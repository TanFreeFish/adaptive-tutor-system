import logging
from app.celery_app import celery_app, get_dynamic_controller
from app.db.database import SessionLocal
from app.schemas.chat import ChatRequest, SocketResponse2, ChatResponse
from app.config.dependency_injection import get_redis_client
from datetime import datetime, timezone
import json
from app.crud.crud_event import event as crud_event
from app.crud.crud_chat_history import chat_history as crud_chat_history
from app.schemas.behavior import BehaviorEvent, EventType, AiHelpRequestData
from app.schemas.chat import ChatHistoryCreate
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def process_chat_request(self, request_data: dict):
    db = SessionLocal()
    try:
        controller = get_dynamic_controller()
        # 将 db 会话传递给需要它的服务方法
        # 调用生成回复（使用同步函数）
        request_obj = ChatRequest(**request_data)
        redis_client = get_redis_client()
        
        # 初始化AI回复缓存和其他分析结果
        ai_response_cache = ""
        sentiment_result = None
        clustering_result = None
        system_prompt = None
        content_title = None
        context_snapshot = None
        
        # stream_start
        message = SocketResponse2(
                type="stream_start",
                taskid=self.request.id,
                timestamp=datetime.now(timezone.utc),
                message="开始",
            )
        redis_client.publish(f"ws:user:{request_data['participant_id']}", message.model_dump_json())
        
        # streaming
        response_generator = controller.generate_adaptive_response_sync(
            request=request_obj,
            db=db,
            background_tasks=None  # Celery任务中不使用FastAPI的BackgroundTasks
        )
        
        for item in response_generator:
            # 如果是元组，说明是最后的分析结果
            if isinstance(item, tuple) and len(item) == 6:
                ai_response_cache, sentiment_result, clustering_result, system_prompt, content_title, context_snapshot = item
            else:
                # 缓存AI回复内容
                ai_response_cache += item
                
                # 注意：响应结果会自动存储在Celery的result backend中
                message = SocketResponse2(
                    type="streaming",
                    taskid=self.request.id,
                    timestamp=datetime.now(timezone.utc),
                    message=item,
                )
                redis_client.publish(f"ws:user:{request_data['participant_id']}", message.model_dump_json())
        
        # stream_end
        message = SocketResponse2(
            type="stream_end",
            taskid=self.request.id,
            timestamp=datetime.now(timezone.utc),
            message="结束",
        )
        redis_client.publish(f"ws:user:{request_data['participant_id']}", message.model_dump_json())
        logger.info(f"写入数据库的数据为{ai_response_cache}")
        # 在流式传输结束后，将完整的AI回复和分析结果记录到数据库
        _log_ai_interaction_in_task(
            request=request_obj,
            ai_response=ai_response_cache,
            db=db,
            sentiment=sentiment_result,
            system_prompt=system_prompt,
            content_title=content_title,
            context_snapshot=context_snapshot
        )
        
    finally:
        db.close()


def _log_ai_interaction_in_task(
    request: ChatRequest,
    ai_response: str,
    db,
    sentiment,
    system_prompt: str = None,
    content_title: str = None,
    context_snapshot: str = None
):
    """
    在Celery任务中记录AI交互。
    """
    try:
        # 准备事件数据
        event = BehaviorEvent(
            participant_id=request.participant_id,
            event_type=EventType.AI_HELP_REQUEST,
            event_data=AiHelpRequestData(message=request.user_message).model_dump(),
            # 统一使用上海时区
            timestamp=datetime.now(ZoneInfo("Asia/Shanghai"))
        )

        # 准备用户聊天记录
        user_chat = ChatHistoryCreate(
            participant_id=request.participant_id,
            role="user",
            message=request.user_message
        )

        # 准备AI聊天记录
        ai_chat = ChatHistoryCreate(
            participant_id=request.participant_id,
            role="assistant",
            message=ai_response,
            raw_prompt_to_llm=system_prompt,
            raw_context_to_llm=context_snapshot + json.dumps(sentiment.model_dump(), ensure_ascii=False) if context_snapshot else json.dumps(sentiment.model_dump(), ensure_ascii=False),
        )

        # 在Celery Worker中，将数据库写入操作作为独立任务分派到db_writer_queue
        # 分派事件记录任务
        celery_app.send_task(
            'app.tasks.db_tasks.log_ai_event_task',
            args=[event.model_dump()], 
            queue='db_writer_queue'
        )
        
        # 分派用户消息记录任务
        celery_app.send_task(
            'app.tasks.db_tasks.save_chat_message_task',
            args=[user_chat.model_dump()], 
            queue='db_writer_queue'
        )
        
        # 分派AI消息记录任务
        celery_app.send_task(
            'app.tasks.db_tasks.save_chat_message_task',
            args=[ai_chat.model_dump()], 
            queue='db_writer_queue'
        )
        
        print(f"INFO: AI interaction for {request.participant_id} queued for async DB write.")
        
    except Exception as e:
        # 数据保存失败必须报错，科研数据完整性优先
        raise RuntimeError(f"Failed to log AI interaction for {request.participant_id}: {e}")
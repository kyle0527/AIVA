#!/bin/bash

# RabbitMQ 死信隊列配置腳本
# 為 AIVA 平台配置統一的死信隊列策略，防止 poison pill 消息問題

set -e

# 配置變數
RABBITMQ_HOST=${AIVA_RABBITMQ_HOST:-localhost}
RABBITMQ_USER=${AIVA_RABBITMQ_USER:-guest}
RABBITMQ_PASSWORD=${AIVA_RABBITMQ_PASSWORD:-guest}
VHOST=${AIVA_RABBITMQ_VHOST:-/}

echo "配置 AIVA 平台的 RabbitMQ 死信隊列策略..."

# 設置所有任務隊列的死信交換機和重試策略
echo "設置任務隊列死信策略..."
rabbitmqctl set_policy \
    --vhost "$VHOST" \
    "aiva-tasks-dlx-policy" \
    "^tasks\." \
    '{
        "dead-letter-exchange": "aiva.dead_letter",
        "dead-letter-routing-key": "failed",
        "message-ttl": 86400000,
        "max-length": 10000
    }' \
    --priority 7 \
    --apply-to queues

# 設置發現隊列的死信策略
echo "設置發現隊列死信策略..."
rabbitmqctl set_policy \
    --vhost "$VHOST" \
    "aiva-findings-dlx-policy" \
    "^findings\." \
    '{
        "dead-letter-exchange": "aiva.dead_letter",
        "dead-letter-routing-key": "failed",
        "message-ttl": 86400000,
        "max-length": 50000
    }' \
    --priority 7 \
    --apply-to queues

# 設置結果隊列的死信策略
echo "設置結果隊列死信策略..."
rabbitmqctl set_policy \
    --vhost "$VHOST" \
    "aiva-results-dlx-policy" \
    "^.*results$" \
    '{
        "dead-letter-exchange": "aiva.dead_letter", 
        "dead-letter-routing-key": "failed",
        "message-ttl": 86400000,
        "max-length": 50000
    }' \
    --priority 7 \
    --apply-to queues

# 創建死信交換機
echo "創建死信交換機..."
rabbitmqctl eval "
    rabbit_exchange:declare(
        resource(<<\"$VHOST\">>, exchange, <<\"aiva.dead_letter\">>),
        topic,
        true,
        false,
        false,
        []
    ).
"

# 創建死信隊列
echo "創建死信隊列..."
rabbitmqctl eval "
    rabbit_amqqueue:declare(
        resource(<<\"$VHOST\">>, queue, <<\"aiva.dead_letter.failed\">>),
        true,
        false,
        [],
        none,
        <<\"admin\">>
    ).
"

# 綁定死信隊列到死信交換機
echo "綁定死信隊列..."
rabbitmqctl eval "
    rabbit_binding:add(
        binding(
            resource(<<\"$VHOST\">>, exchange, <<\"aiva.dead_letter\">>),
            <<\"failed\">>,
            resource(<<\"$VHOST\">>, queue, <<\"aiva.dead_letter.failed\">>),
            []
        )
    ).
"

echo "✅ RabbitMQ 死信隊列配置完成！"
echo ""
echo "配置摘要："
echo "- 死信交換機: aiva.dead_letter"
echo "- 死信隊列: aiva.dead_letter.failed"
echo "- 消息 TTL: 24 小時"
echo "- 適用於所有 tasks.*, findings.*, *results 隊列"
echo ""
echo "查看配置："
echo "  rabbitmqctl list_policies --vhost $VHOST"
echo "  rabbitmqctl list_exchanges --vhost $VHOST | grep dead_letter"
echo "  rabbitmqctl list_queues --vhost $VHOST | grep dead_letter"
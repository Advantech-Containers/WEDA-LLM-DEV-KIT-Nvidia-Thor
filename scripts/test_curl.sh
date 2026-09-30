#!/usr/bin/env bash
# ==============================================================================
# WEDA-LLM-Cognitive-RAG — Quick Test Script
# ==============================================================================

# Endpoint follows .env
# script. Always loopback: the API is bound to 127.0.0.1 and this runs on-device.
PORT="${WEDA_LLM_PORT:-9000}"
API="http://127.0.0.1:${PORT}"

# Fetch the active served model name dynamically from the vLLM endpoint
MODELS_JSON=$(curl -s "${API}/v1/models" 2>/dev/null)
MODEL_ID=$(python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
if models:
    print(models[0].get('id', ''))
" <<< "$MODELS_JSON" 2>/dev/null)

if [ -z "$MODEL_ID" ]; then
    echo "[ERROR] No active model found. Ensure the serving container is running."
    exit 1
fi

QUESTION="How does deploying a local, optimized Cognitive RAG container on an edge gateway benefit industrial operations compared to cloud-based LLM services? Explain in one brief, professional, and impactful sentence."

echo -e "\033[1;36m============================================================\033[0m"
echo -e "\033[1;36m    WEDA-LLM-Cognitive-RAG — Local Edge Query\033[0m"
echo -e "\033[1;36m============================================================\033[0m"
echo -e "  \033[1;33mQuestion:\033[0m"
echo -e "  \"$QUESTION\"\n"

echo "Querying model $MODEL_ID (waiting for response)..."

T_START=$(date +%s%N)
RESPONSE=$(curl -s -X POST "${API}/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "'"$MODEL_ID"'",
       "messages": [{"role": "user", "content": "'"$QUESTION"'"}],
       "max_tokens": 1024
     }')
T_END=$(date +%s%N)
ELAPSED_MS=$(python3 -c "print(f'{($T_END - $T_START) / 1e6:.1f}')" 2>/dev/null)

# Print response content and execution details dynamically using python via stdin
python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    elapsed_ms = float(sys.argv[1])

    msg = data.get('choices', [{}])[0].get('message', {})
    content = msg.get('content')
    reasoning = msg.get('reasoning')
    model = data.get('model', 'N/A')

    usage = data.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)

    # Calculate tokens/sec (decode)
    if completion_tokens > 0 and elapsed_ms > 0:
        tokens_per_sec = f'{completion_tokens / (elapsed_ms / 1000.0):.2f} tok/s'
    else:
        tokens_per_sec = 'N/A'

    print('\n\033[1;32m  Response:\033[0m')
    if content and content.strip() and content != 'None':
        print(f'  {content.strip()}')
    else:
        print('  [No content generated — model output is empty or reasoning-only]')

    if reasoning and reasoning.strip():
        print('\n\033[1;35m============================================================\033[0m')
        print('\033[1;35m    Model Reasoning / Chain of Thought\033[0m')
        print('\033[1;35m============================================================\033[0m')
        for line in reasoning.strip().split('\n'):
            print(f'  {line}')

    print('\n\033[1;36m============================================================\033[0m')
    print('\033[1;36m    Execution & Technical Metadata\033[0m')
    print('\033[1;36m============================================================\033[0m')
    print(f'  Served Model:         {model}')
    print(f'  Prompt Tokens:        {prompt_tokens}')
    print(f'  Completion Tokens:    {completion_tokens}')
    print(f'  Total Tokens:         {total_tokens}')
    print(f'  Total Latency:        {elapsed_ms:.1f} ms ({elapsed_ms/1000.0:.2f} s)')
    print(f'  Throughput (decode):  {tokens_per_sec}')
    print('\033[1;36m============================================================\033[0m')
except Exception as e:
    print(f'\n[ERROR] Failed to parse JSON response: {e}')
" "$ELAPSED_MS" <<< "$RESPONSE"

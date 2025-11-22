#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# agent_gen.sh
#
# Purpose
# -------
#   A lightweight chat‑assistant wrapper that can talk to two LLM providers:
#
#   1️⃣  Hugging Face router (`openai/gpt‑oss:latest`) – accessed via
#       https://router.huggingface.co/v1/chat/completions
#
#   2️⃣  A local Ollama instance that already exposes the chat endpoint at
#       http://localhost:11434/api/chat
#
#   The script also demonstrates *function‑calling* (a new OpenAI‑style
#   feature) with two tiny tools:
#       • call_cowsay  – generate a cowsay‑styled speech bubble
#       • get_cow_files – list the cow‑art files available on the host
#
#   The demo at the bottom runs a short conversation that calls those tools.
#
# Usage
# -----
#   Prerequisites:
#       • Bash  (>= 4 for associative arrays)
#       • jq   – for JSON creation/parsing
#       • cowsay (optional – required only for the tools)
#       • Either:
#           – a running Ollama server on localhost:11434, or
#           – a Hugging Face router token in $OPENAI_TOKEN
#
#   Run the script (the demo is executed automatically):
#       $ ./agent_gen.sh
#
#   To force a particular provider, export the variable before running:
#       $ LLM_PROVIDER=openai   # uses the router
#       $ LLM_PROVIDER=ollama   # uses the local instance
#
#   Enable debugging (extra payload/response dumps) by setting:
#       $ DEBUG=1
#
#   Optional arguments:
#       * test          – run the simple unit‑test instead of the demo
#       * interactive   – start a REPL where you can type in your own prompts
#
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fail fast – any unset variable, any non‑zero exit status, or any pipe error
# will cause the script to abort immediately.  This prevents silent failures.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Ollama already includes the full /api/chat suffix, so we store the whole URL.
OLLAMA_HOST="http://localhost:11434/api/chat"

# OpenAI compatible base URL (the “v1” part is required by the API).
OPENAI_HOST="https://router.huggingface.co/v1"

# Default LLM provider.  The guard after this will automatically switch to
# Ollama if we ask for OpenAI but have no token.
LLM_PROVIDER="ollama"

# ---------------------------------------------------------------------------
# Provider‑selection guard
# ---------------------------------------------------------------------------
# If the user requested the OpenAI router *and* has not exported an API token,
# fall back to the local Ollama instance.  The `${OPENAI_TOKEN:-}` expansion
# yields an empty string when the variable is missing – safe under `set -u`.
if [[ "$LLM_PROVIDER" == "openai" && -z "${OPENAI_TOKEN:-}" ]]; then
    echo "Variable OPENAI_TOKEN is not set – switching to Ollama" >&2
    LLM_PROVIDER="ollama"
fi

# ---------------------------------------------------------------------------
# Model name to request from the provider.
# ---------------------------------------------------------------------------
# Uncomment the longer model if you need a larger one.
# MODEL="openai/gpt-oss-120b:novita"
MODEL="gpt-oss:latest"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
declare -i _tool_depth=0                # recursion depth for tool calls
MAX_TOOL_RECURSION=8
declare -a MESSAGES                     # Bash array containing JSON strings.
DEBUG=false                             # false = no debug output, true = dump payload/response.

# ---------------------------------------------------------------------------
# JSON helper functions – they use `jq` to guarantee proper escaping.
# ---------------------------------------------------------------------------
create_message() {
    # $1 – message content.
    # $2 – role (defaults to "user").
    local content="$1"
    local role="${2:-user}"
    jq -nc --arg role "$role" --arg content "$content" \
        '{"role":$role,"content":$content}'
}

create_tool_message() {
    # $1 – name of the tool that generated the result.
    # $2 – textual output of the tool.
    # $3 – tool‑call ID that the model sent.
    local function_name="$1"
    local content="$2"
    local call_id="$3"
    jq -n --arg name "$function_name" \
          --arg content "$content" \
          --arg tool_call_id "$call_id" \
          '{"role":"tool","tool_call_id":$tool_call_id,"name":$name,"content":$content}'
}

# ---------------------------------------------------------------------------
# Tool implementations – the *real* work the model can request.
# ---------------------------------------------------------------------------
call_cowsay() {
    # $1 – the text that the cow should say.
    # $2 – optional cow‑art file name.
    local message="$1"
    if [[ -z "$message" ]]; then
        echo "Message for cow missing, please provide one."
        return 1
    fi
    local cow="${2:-default}"
    cowsay -f "$cow" "$message"
}

get_cow_files() {
    # `cowsay -l` prints a space‑separated list; convert to one name per line.
    cowsay -l | tail -n +2 | tr ' ' '\n'
}

# ---------------------------------------------------------------------------
# Tool‑call handling – safely process any number of calls that the model
# may request.  The function receives the raw JSON array under `tool_calls`.
# ---------------------------------------------------------------------------
process_tool_calls() {
    local tool_call_json="$1"

    # If the field is null or an empty array, there is nothing to do.
    if [[ "$tool_call_json" == "null" || "$tool_call_json" == "[]" ]]; then
        $DEBUG && echo "No tool calls present." >&2
        _tool_depth=0
        return
    fi

    # Extract the first call – the demo only ever sends a single call.
    local call_id func
    call_id=$(printf '%s' "$tool_call_json" | jq -r '.[0].id // empty')
    func=$(printf '%s' "$tool_call_json" | jq -r '.[0].function.name // empty')
    printf "Processing tool call %s → %s\n" "$call_id" "$func"

    case "$func" in
        call_cowsay)
            local message body result tmsg
            body=$(printf '%s' "$tool_call_json" | jq -r '.[0].function.arguments.file // empty')
            message=$(printf '%s' "$tool_call_json" | jq -r '.[0].function.arguments.message // empty')
            result=$(call_cowsay "$message" "$body")
            tmsg=$(create_tool_message call_cowsay "$result" "$call_id")
            add_message "$tmsg"
            send_message
            ;;
        get_cow_files)
            local cows tmsg
            cows=$(get_cow_files)
            tmsg=$(create_tool_message get_cow_files "$cows" "$call_id")
            add_message "$tmsg"
            send_message
            ;;
        '')
            # Empty – ignore.
            ;;
        *)
            echo "Warning: unknown tool function '$func' – ignored" >&2
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Message handling – turn a raw JSON message into human‑readable console output.
# ---------------------------------------------------------------------------
read_message() {
    local msg="$1"
    local role content tool_calls

    role=$(printf '%s' "$msg" | jq -r '.role')
    content=$(printf '%s' "$msg" | jq -r '.content')
    tool_calls=$(printf '%s' "$msg" | jq -r '.tool_calls')

    # If the assistant returned tool calls, handle them now.
    process_tool_calls "$tool_calls"

    # Show the textual part of the message (if any).
    if [[ -n "$content" && "$role" != 'tool' ]]; then
        printf '[%s]: %s\n' "$role" "$content"
    fi    
}

add_message() {
    # Append a JSON string to the conversation history and (optionally) preview it.
    local message="$1"
    $DEBUG && echo "Adding message '$message'"
    read_message "$message" 2>/dev/null
    MESSAGES+=("$message")
}

# ---------------------------------------------------------------------------
# Tool definition – JSON schema we send to the model so it knows which
# functions it may call.
# ---------------------------------------------------------------------------
get_tools() {
    # The JSON blocks contain back‑ticks; we keep them inside single‑quotes
    # so the shell does not try to interpret them.
    local call_cowsay_tool='{
        "type":"function",
        "function":{
            "name":"call_cowsay",
            "description":"Render a cowsay‑style speech bubble with the supplied text (and optionally a cow‑art file) and return the formatted string to the caller.",
            "parameters":{
                "type":"object",
                "properties":{
                    "message":{"type":"string","description":"The text the cow should display inside its speech bubble."},
                    "file":{"type":"string","description":"Name of a cow‑art file to use for rendering. Valid file names are obtained by calling the `get_cow_files` function (no parameters)."}
                },
                "required":["message"]
            }
        }
    }'

    local get_cow_files_tool='{
        "type":"function",
        "function":{
            "name":"get_cow_files",
            "description":"Return a list of cow‑art file names that are available on the host system for use with the `call_cowsay` function.",
            "parameters":{"type":"object","properties":{},"required":[]}
        }
    }'

    # Produce a JSON array that contains the two tool objects.
    printf '%s\n' "$call_cowsay_tool" "$get_cow_files_tool" | jq -s '.'
}

# ---------------------------------------------------------------------------
# Payload construction – the JSON body that gets POSTed to the provider.
# ---------------------------------------------------------------------------
create_payload() {
    local messages_json tools_json
    messages_json=$(printf '%s\n' "${MESSAGES[@]}" | jq -s '.')   # array of message objects
    tools_json=$(get_tools)

    # Ollama needs `response_format:{type:"json_object"}` to force JSON output.
    if [[ $LLM_PROVIDER == ollama ]]; then
    jq -n \
        --arg model "$MODEL" \
        --argjson tools "$tools_json" \
        --argjson messages "$messages_json" \
            '{
                model:$model,
                stream:false,
                tools:$tools,
                tool_choice:"auto",
              response_format:{type:"json_object"},
                messages:$messages
            }'
    else
        # Hugging‑Face router (OpenAI‑compatible) does not understand
        # `response_format`.
        jq -n \
            --arg model "$MODEL" \
            --argjson tools "$tools_json" \
            --argjson messages "$messages_json" \
            '{
                model:$model,
                stream:false,
                tools:$tools,
                tool_choice:"auto",
                messages:$messages
            }'
    fi
}

# ---------------------------------------------------------------------------
# Response extraction – turn the raw HTTP body into a single message object.
# ---------------------------------------------------------------------------
get_message_from_openai_response() {
    local response="$1"
    local message
    message=$(printf '%s' "$response" | jq -c '.choices[0].message')
    add_message "$message"
}

get_message_from_ollama_response() {
    local response="$1"
    local message
    message=$(printf '%s' "$response" | jq -c '.message')
    add_message "$message"
}

# Dispatcher that chooses the correct extractor based on $LLM_PROVIDER.
get_message_from_response() {
    local response="$1"
    case "$LLM_PROVIDER" in
        openai) get_message_from_openai_response "$response" ;;
        ollama) get_message_from_ollama_response "$response" ;;
        *) echo "Unsupported LLM provider: $LLM_PROVIDER" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Provider‑specific send functions – build the payload, POST it, then hand the
# raw response to the extractor above.
# ---------------------------------------------------------------------------
openai_send_message() {
    local payload response
    payload=$(create_payload)

    $DEBUG && echo "---- PAYLOAD ----" >&2
    $DEBUG && printf '%s\n' "$payload" | jq . >&2

    local endpoint="${OPENAI_HOST}/chat/completions"
    response=$(curl -sS "$endpoint" \
        -H "Authorization: Bearer ${OPENAI_TOKEN}" \
        -H 'Content-Type: application/json' \
        --data "$payload")

    $DEBUG && echo "---- RESPONSE ----" >&2
    $DEBUG && printf '%s\n' "$response" | jq . >&2

    get_message_from_response "$response"
}

ollama_send_message() {
    local payload response
    payload=$(create_payload)

    $DEBUG && echo "---- PAYLOAD ----" >&2
    $DEBUG && printf '%s\n' "$payload" | jq . >&2

    # $OLLAMA_HOST already ends with /api/chat – do *not* add another suffix.
    response=$(curl -sS "$OLLAMA_HOST" \
        -H 'Content-Type: application/json' \
        --data "$payload")

    $DEBUG && echo "---- RESPONSE ----" >&2
    $DEBUG && printf '%s\n' "$response" | jq . >&2

    get_message_from_response "$response"
}

# ---------------------------------------------------------------------------
# Unified entry point – picks the right provider based on $LLM_PROVIDER.
# ---------------------------------------------------------------------------
send_message() {
    case "$LLM_PROVIDER" in
        openai) openai_send_message ;;
        ollama) ollama_send_message ;;
        *) echo "Unsupported LLM provider: $LLM_PROVIDER" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Demo / test harness – runs a three‑step conversation.
# ---------------------------------------------------------------------------
QUESTIONS=(
    'Which cow bodies my system support?'
    'Tell me a joke, use cowsay.'
    'Use kangaroo and tell me one more.'
)

make_chat() {
    # Fresh conversation each run.
    MESSAGES=()

    # System prompt – must be the first message.
    local sys_msg
    sys_msg=$(create_message "You are an AI assistant with tool support" "system")
    add_message "$sys_msg"

    for QUESTION in "${QUESTIONS[@]}"; do
        local user_msg
        user_msg=$(create_message "$QUESTION")
        add_message "$user_msg"
        send_message
    done
}

# ---------------------------------------------------------------------------
# Small unit‑test – feeds a handcrafted assistant message that contains a mock
# tool call. The handler will execute the tool and emit its output.
# ---------------------------------------------------------------------------
test() {
    local msg='{"role": "assistant", "content": "", "thinking": "We need to produce a joke using cowsay. We should call the function call_cowsay. The user didn'\''t specify a particular cow, so we can choose default or something. The cow must be one of the files above. Let'\''s pick \"default\" or something. We'\''ll create a joke string. Then use function.", "tool_calls": [ { "id": "call_tj0y4y59", "function": { "index": 0, "name": "call_cowsay", "arguments": { "file": "default", "message": "Why don'\''t programmers like nature? Too many bugs!"}}}]}'
    # printf "%s" "$msg"  | jq -r '.tool_calls[].function.arguments.file // empty'
    read_message "$msg"
}
# test

# ---------------------------------------------------------------------------
# Interactive REPL – type a prompt, get a reply, repeat until EOF or “exit”.
# ---------------------------------------------------------------------------
repl() {
    MESSAGES=()
    # System prompt.
    add_message "$(create_message "You are an AI assistant with tool support" "system")"

    echo "Enter your prompts (type 'exit' or press Ctrl‑D to quit):"
    while true; do
        printf '> '
        if ! IFS= read -r user_input; then
            echo   # newline after Ctrl‑D
            break
        fi
        [[ "$user_input" == "exit" ]] && break
        add_message "$(create_message "$user_input")"
        send_message
    done
}

# ---------------------------------------------------------------------------
# Main entry point – decide what to run based on optional command‑line args.
# ---------------------------------------------------------------------------
case "${1:-}" in
    test)
        test
        ;;
    interactive)
        repl
        ;;
    "")
        # No argument – run the built‑in demo.
        make_chat
        ;;
    *)
        echo "Usage: $0 [test|interactive]" >&2
        exit 1
        ;;
esac

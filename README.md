# transformers-chat-api

```bash
curl -s http://192.168.0.77:8000/v1/completions/ \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "What you know about sun?",
        "max_tokens": 60,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop": "eof"
    }' 
```

```bash
curl -s http://192.168.0.77:8000/v1/chat/completions/ \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What you know about sun?"}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 120
    }'
```

```bash
curl http://192.168.0.77:8000/v1/images/generations/ \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-image-1",
        "prompt": "cat sitting on a chair",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json"
    }'
```

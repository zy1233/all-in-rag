curl --request POST --url https://api.siliconflow.cn/v1/chat/completions \
  --header 'Authorization: Bearer sk-bizuvogbjxzspfcxjxgkuuqolbhcetxftwpxdczumjfhiarw' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "deepseek-ai/DeepSeek-V3",
  "messages": [
    {
      "role":"system",
      "content": "使用中文回答问题，要简明扼要",
      "role": "user",
      "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
    }
  ]
}'
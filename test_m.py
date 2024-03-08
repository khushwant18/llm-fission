import requests, time
prompt="cat sat on the"
url="http://0.0.0.0:7002/generate"

start = time.time()
res = requests.post(url, json={"prompt": prompt,"max_len":10})
print(time.time()-start)
print(res.text)
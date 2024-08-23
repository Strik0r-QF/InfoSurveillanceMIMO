# 获取当前时间
from datetime import datetime

now = datetime.now()

# 格式化时间
formatted_time = now.strftime('%Y/%m/%d-%H:%M')
print(formatted_time)
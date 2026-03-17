# Bridge 从 0 到 1 部署说明

## 作用
Bridge 负责：
1. 下载用户上传的产品图
2. 上传到 ComfyUI Cloud
3. 用 API-format workflow 提交任务
4. 轮询任务完成状态
5. 取回输出文件
6. 生成稳定的 `image_url`
7. 生成 16 张候选图并筛到 12 张

## 最少环境变量
见 `.env.example`

## 本地启动
```bash
npm install
npm start
```

## 生产部署
任何支持 Node 20 的环境都可以。
要求：
- 有公网 HTTPS
- 能读取环境变量
- 能访问 `https://cloud.comfy.org`

## 可用接口
### 1. GET /health
检查服务是否存活。

### 2. POST /generate_one
单张生成，适合调试。

### 3. POST /run_batch16
主接口。Coze 主流程建议只调用这个接口。

## 为什么推荐 run_batch16
因为这样 Coze 侧不需要 16 次 Loop，也不需要 16 次插件调用。
对新手最稳、最省事。

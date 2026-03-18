# NLM Backend

Ye folder Railway aur Render dono ke liye ready hai.

## Render setup

1. Agar `backend` ko alag repo `lmsBackend` bana rahe ho to is folder ka content repo root me push karo.
2. Render me `Web Service` banao.
3. Runtime `Python 3` rakho.
4. Build command:

`pip install -r requirements.txt`

5. Start command:

`uvicorn main:app --host 0.0.0.0 --port $PORT`

6. Health check path:

`/api/health`

7. Environment variables set karo:
   - `GROQ_API_KEY`
   - `FRONTEND_URL=https://lmsprototype-six.vercel.app`
   - `CORS_ORIGINS=https://lmsprototype-six.vercel.app`
   - `PYTHON_VERSION=3.11.11`

## Railway setup

1. GitHub repo `lmsBackend` me is folder ka content push karo.
2. Railway me new project banao aur us repo ko connect karo.
3. Environment variables set karo:
   - `GROQ_API_KEY`
   - `FRONTEND_URL=https://lmsprototype-six.vercel.app`
   - `CORS_ORIGINS=https://lmsprototype-six.vercel.app`
4. Deploy ke baad `https://your-backend-domain.up.railway.app/api/health` open karke check karo.

## Frontend setup

Vercel me `VITE_API_BASE_URL` ko apne backend URL par set karo, for example:

`VITE_API_BASE_URL=https://your-backend-domain.onrender.com`

ya

`VITE_API_BASE_URL=https://your-backend-domain.up.railway.app`

Frontend already `/api/chat`, `/api/reset/:sessionId`, aur `/api/resume/analyze` endpoints use karta hai.

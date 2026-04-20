# Halacha Chat UI

דף צ'אט בעברית לשאלות בהלכה, עם קריאה ל-OpenAI דרך פונקציית שרת.

🔗 **דמו חי:** [halacha-chat.vercel.app](https://halacha-chat.vercel.app)

## הרצה מקומית

**1. שכפל את הריפו**
```bash
git clone https://github.com/egozi/rag_shul.git
cd rag_shul/halacha-chat-ui
```

**2. צור קובץ `.env`**
```bash
cp .env.example .env
```
פתח את `.env` והכנס את המפתח שלך:
```
OPENAI_API_KEY=sk-...
```
מפתח ניתן לקבל בחינם (עם מכסה) ב-[platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**3. התקן והרץ**
```bash
pip install -r requirements.txt
vercel dev
```
פתח את הדפדפן על `http://localhost:3000`

## פריסה ל-Vercel

1. צור פרויקט חדש ב-[vercel.com](https://vercel.com)
2. ב-**Settings → Environment Variables** הוסף את `OPENAI_API_KEY`
3. הרץ `vercel --prod`

## קבצים

| קובץ | תפקיד |
|------|--------|
| `index.html` | ממשק הצ'אט |
| `api/chat.py` | שרת Python שמדבר עם OpenAI |
| `requirements.txt` | תלויות Python |
| `.env.example` | תבנית לקובץ הסביבה |
| `vercel.json` | הגדרות Vercel |

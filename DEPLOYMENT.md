# Deployment Guide

## GitHub Setup
1. Repository is configured for: `https://github.com/AtharvaLotankar11/Student-Performance-Predictor---Data-Mining-GROQ-API.git`

## Vercel Deployment Steps

### 1. Prerequisites
- Vercel account (sign up at vercel.com)
- GROQ API key (get from console.groq.com)

### 2. Deploy to Vercel
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Configure environment variables:
   - Add `GROQ_API_KEY` with your API key value
5. Deploy!

### 3. Important Notes
- The dataset `StudentsPerformance.csv` needs to be included in the repository
- Models will be trained on first deployment (may take time)
- Vercel has memory and execution time limits for free tier

### 4. Local Development
```bash
pip install -r requirements.txt
python app.py
```

### 5. Environment Variables
Set these in Vercel dashboard:
- `GROQ_API_KEY`: Your GROQ API key for AI insights
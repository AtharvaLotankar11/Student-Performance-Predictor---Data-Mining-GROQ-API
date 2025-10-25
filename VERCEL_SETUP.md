# Vercel Deployment Guide

## 🚀 Quick Deploy to Vercel

### Step 1: Get Your GROQ API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up/Login
3. Create a new API key
4. Copy the key (you'll need it for Vercel)

### Step 2: Deploy to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click "New Project"
4. Import this repository: `AtharvaLotankar11/Student-Performance-Predictor---Data-Mining-GROQ-API`
5. Configure the project:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: (leave empty)
   - Output Directory: (leave empty)

### Step 3: Add Environment Variables
In Vercel dashboard:
1. Go to Settings → Environment Variables
2. Add: `GROQ_API_KEY` = `your_groq_api_key_here`
3. Save

### Step 4: Deploy
1. Click "Deploy"
2. Wait for deployment (first deploy may take 5-10 minutes due to ML model training)
3. Your app will be live at `https://your-project-name.vercel.app`

## ⚠️ Important Notes

### Memory Limitations
- Vercel free tier has memory limits
- If deployment fails due to memory, consider:
  - Using smaller ML models
  - Reducing dataset size
  - Upgrading to Vercel Pro

### Cold Starts
- First request after inactivity may be slow (model loading)
- Subsequent requests will be faster

### File Size
- The dataset file is included in the repository
- Total deployment size should stay under Vercel limits

## 🔧 Troubleshooting

### Common Issues:
1. **Memory Error**: Reduce model complexity in `app.py`
2. **Timeout**: Increase function timeout in `vercel.json`
3. **API Key Error**: Check environment variable spelling

### Local Testing:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
set GROQ_API_KEY=your_key_here

# Run locally
python app.py
```

## 📱 Features Available After Deployment:
- Complete student performance prediction
- Multiple ML models (Decision Tree, Random Forest, SVM, Neural Network)
- AI-powered insights via GROQ API
- Interactive web interface
- Real-time predictions

Your app will be accessible worldwide once deployed! 🌍
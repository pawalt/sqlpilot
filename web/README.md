# SQLPilot Web UI

A static web interface for SQL autocomplete using Transformers.js.

## Quick Start (with Modal)

If you trained your model with Modal, this is the easiest path:

```bash
# 1. Export model to ONNX (runs on Modal)
modal run modal_train.py::export_onnx --model-size 15M

# 2. Download ONNX model to web/model/
modal run modal_train.py::download_onnx_local --model-size 15M

# 3. Serve the web UI
cd web
python -m http.server 8080
```

Open http://localhost:8080

## Manual ONNX Export (without Modal)

If you have a local PyTorch checkpoint:

### 1. Install Optimum

```bash
pip install optimum[exporters] onnx onnxruntime
```

### 2. Export to ONNX

```bash
# Export the model
optimum-cli export onnx \
    --model ./downloaded_model \
    --task text-generation \
    ./web/model
```

### 3. Copy Tokenizer Files

The tokenizer files are needed alongside the ONNX model:

```bash
# Copy tokenizer files to model directory
cp ./downloaded_model/tokenizer.json ./web/model/
cp ./downloaded_model/tokenizer_config.json ./web/model/
cp ./downloaded_model/special_tokens_map.json ./web/model/
cp ./downloaded_model/config.json ./web/model/
```

### 4. Verify Directory Structure

Your `web/model/` directory should contain:

```
web/model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json  # (optional)
└── onnx/
    └── model_quantized.onnx  # Transformers.js looks for this name
```

**Note:** Transformers.js expects the ONNX file at `onnx/model_quantized.onnx`. If your export creates `model.onnx`, move it:
```bash
mkdir -p web/model/onnx
mv web/model/model.onnx web/model/onnx/model_quantized.onnx
```

### 5. Test Locally

```bash
cd web
python -m http.server 8080
# Open http://localhost:8080
```

## Deploying to Production

This is a fully static site - deploy to any static hosting:
- GitHub Pages
- Netlify
- Vercel
- Cloudflare Pages
- S3 + CloudFront
- Your own server with nginx

Just upload the entire `web/` directory including the `model/` folder.

**Note:** Model files can be large (50-200MB). Some hosts have file size limits.

## Alternative: HuggingFace Hub

If you prefer to host the model on HuggingFace Hub instead:

1. Upload the ONNX model to HuggingFace Hub
2. In `index.html`, change:
   ```javascript
   const MODEL_PATH = 'your-username/sqlpilot-15m';
   env.allowLocalModels = false;
   ```

## Performance Tips

1. **WebGPU**: The UI automatically tries WebGPU first for faster inference
2. **Model Size**: Smaller models (15M params) work better in browsers
3. **Caching**: Models are cached in IndexedDB after first load
4. **Compression**: Consider gzip/brotli compression on your server for faster downloads

## Troubleshooting

### Model doesn't load
- Check browser console for errors (F12)
- Verify ONNX files exist in `web/model/`
- Ensure `tokenizer.json` and `config.json` are present
- Check CORS headers if serving from different domain

### CORS errors with `python -m http.server`
The simple Python server should work for same-origin requests. If you see CORS errors, try:
```bash
# Use a CORS-enabled server
npx serve -p 8080
```

### Slow performance
- Try Chrome/Edge (better WebGPU support)
- First load downloads the model; subsequent loads use cache
- Generation takes 1-3 seconds on CPU, faster on WebGPU

### Suggestions are wrong
- Check that table schema format matches training data
- Model expects `### TABLEDATA` and `### STATEMENT` markers


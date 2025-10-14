[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask3&demo-title=Flask%203%20%2B%20Vercel&demo-description=Use%20Flask%203%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask3-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)

# Flask + Vercel

This example shows how to use Flask 3 on Vercel with Serverless Functions using the [Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python).

## Demo

https://flask-python-template.vercel.app/

## How it Works

This example uses the Web Server Gateway Interface (WSGI) with Flask to enable handling requests on Vercel with Serverless Functions.

## Running Locally

```bash
npm i -g vercel
vercel dev
```

Your Flask application is now available at `http://localhost:3000`.

## One-Click Deploy

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=vercel-examples):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask3&demo-title=Flask%203%20%2B%20Vercel&demo-description=Use%20Flask%203%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask3-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)

## Integrity Analysis API (Gemini-powered)

Endpoint: `POST /analyze`

Body (JSON):

```json
{
  "baseline_url": "https://example.com/baseline.jpg",
  "current_url": "https://example.com/current.jpg"
}
```

Alternatively, you can send base64:

```json
{
  "baseline_b64": "data:image/jpeg;base64,/9j/...",
  "current_b64": "data:image/jpeg;base64,/9j/..."
}
```

Response (JSON) conforms to the requested schema, for example:

```json
{
  "differences": [
    {
      "id": "diff-1",
      "region": "top-right corner",
      "bbox": [0.72, 0.06, 0.18, 0.2],
      "type": "dent",
      "description": "Visible dent with local contour collapse.",
      "severity": "MEDIUM",
      "confidence": 0.82,
      "explainability": [
        "edge discontinuity",
        "local shading change",
        "92% patch difference"
      ],
      "suggested_action": "Supervisor review",
      "tis_delta": -12
    }
  ],
  "baseline_image_info": {
    "resolution": [1024, 768],
    "exif_present": true,
    "camera_make": "...",
    "camera_model": "...",
    "datetime": "..."
  },
  "current_image_info": {
    "resolution": [1024, 768],
    "exif_present": true,
    "camera_make": "...",
    "camera_model": "...",
    "datetime": "..."
  },
  "aggregate_tis": 78,
  "overall_assessment": "REVIEW_REQUIRED",
  "confidence_overall": 0.86,
  "notes": "Detected potential integrity issues; follow suggested actions."
}
```

### Environment Variables

Set your Gemini API key in Vercel project settings (or locally):

- `GOOGLE_API_KEY` (preferred) or `GEMINI_API_KEY`

### Local Test

```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_url": "https://upload.wikimedia.org/wikipedia/commons/7/77/Delete_key1.jpg",
    "current_url": "https://upload.wikimedia.org/wikipedia/commons/7/77/Delete_key1.jpg"
  }'
```

If the API key is set and images differ, the endpoint will return detected differences and Trust Integrity Score (TIS). If the key is not set, the endpoint returns an empty `differences` array and `aggregate_tis: 100`.

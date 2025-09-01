# Gardener Service 🌱

A FastAPI service that creates an "idea garden" by automatically clustering and organizing content from LibreChat AI conversations for RAG retrieval.

## Features

- **Automatic Topic Clustering**: Uses semantic embeddings to group related conversation snippets
- **RAG Retrieval**: Hybrid search combining topic and query similarity
- **Real-time Processing**: Processes LibreChat conversation turns as they happen
- **Flexible Storage**: Supports both PostgreSQL and in-memory fallback
- **Multiple Embedding Options**: OpenAI or SentenceTransformer models

## API Endpoints

- `POST /turn` - Process conversation turns from LibreChat
- `GET /topics/search?q={query}&k={limit}` - Search topics semantically
- `GET /topics/{topic_id}/brief` - Get topic summary with top snippets
- `POST /retrieve` - RAG retrieval with hybrid scoring
- `GET /topics` - List all topics with metadata
- `GET /health` - Health check and system status

## Environment Variables

```bash
# Required
TOPICNAV_TOKEN=your_secret_token_here

# Database (Railway provides automatically)
DATABASE_URL=postgresql://user:password@host:port/database

# Embedding Configuration
USE_OPENAI=false                    # Set to true for OpenAI embeddings
OPENAI_API_KEY=your_openai_key     # Required if USE_OPENAI=true
EMBEDDING_MODEL=all-MiniLM-L6-v2   # SentenceTransformer model name

# Railway sets this automatically
PORT=8000
```

## Railway Deployment

This service is designed to integrate with your existing Railway project containing LibreChat, VectorDB, and MongoDB.

1. **Connect to Railway**: Add this as a new service in your existing Railway project
2. **Environment Variables**: Railway will auto-provide `DATABASE_URL` and `PORT`
3. **Set Required Variables**: Add `TOPICNAV_TOKEN` and optionally `OPENAI_API_KEY`
4. **Deploy**: Railway will automatically build and deploy using the Dockerfile

## Integration with LibreChat

Configure LibreChat to send conversation data to this service:

```bash
# In LibreChat environment
GARDENER_SERVICE_URL=https://your-gardener-service.railway.app
GARDENER_TOKEN=your_topicnav_token
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run the service
uvicorn app:app --reload --port 8000
```

## Architecture

- **Topics**: Semantic clusters with dynamic centroids updated via EMA
- **Snippets**: Individual text fragments with embeddings and topic assignments
- **Similarity Threshold**: 0.78 for topic assignment (creates new topic if below)
- **Hybrid Scoring**: 60% topic similarity + 40% query similarity for retrieval

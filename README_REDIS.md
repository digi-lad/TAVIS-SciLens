# Scaling with Redis (Future Enhancement)

If you reach 50+ concurrent users and need horizontal scaling, add Redis for SocketIO clustering.

## Setup

1. **Add Redis dependency:**
```bash
pip install redis
```

2. **Update app.py:**
```python
import redis
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    message_queue='redis://your-redis-url:6379'
)
```

3. **Get Redis URL:**
- Sign up at https://upstash.com (free tier available)
- Or use Redis Cloud (free tier: 30MB)
- Copy the Redis connection URL

4. **Deploy multiple Render instances:**
- In Render, scale to 2-3 instances
- All instances connect to same Redis
- SocketIO messages broadcast across all instances

## When to Scale
- **Current (Waitress)**: Up to 50-100 concurrent users
- **With Redis**: 10,000+ concurrent users (across multiple instances)

## Cost Estimate
- Render: $7/month per instance
- Upstash Redis: Free (10K commands/day) or $10/month unlimited
- **Total for 3 servers**: ~$21/month



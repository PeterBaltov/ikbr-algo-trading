# moneytrailz Dashboard - Production Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- SSL certificates (optional but recommended)
- Domain name configured
- Minimum 4GB RAM, 20GB disk space

### 1. Environment Setup

Create a `.env.production` file in the root directory:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password_here
DATABASE_URL=postgresql://moneytrailz:your_secure_postgres_password_here@database:5432/thetagang_dashboard

# Application Security
SECRET_KEY=your_super_secure_secret_key_here_at_least_32_characters_long
JWT_SECRET_KEY=another_super_secure_jwt_secret_key_here

# API Configuration
NEXT_PUBLIC_API_URL=https://api.your-domain.com
NEXT_PUBLIC_WS_URL=wss://api.your-domain.com

# CORS Settings
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password_here

# Feature Flags
ENABLE_MONITORING=true
ENABLE_SSL=true
```

### 2. SSL Certificate Setup (Recommended)

Create SSL certificates directory:
```bash
mkdir -p nginx/ssl
```

Option A: Self-signed certificates (development)
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem
```

Option B: Let's Encrypt (production)
```bash
# Install certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
```

### 3. Deploy

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

### 4. Access the Application

- **Dashboard**: https://your-domain.com
- **API Documentation**: https://your-domain.com/api/docs
- **Grafana Monitoring**: https://your-domain.com:3001
- **Prometheus Metrics**: https://your-domain.com:9090

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Nginx     │    │   Frontend  │    │   Backend   │
│ (Port 80/443)│──▷ │ (Port 3000) │ ──▷│ (Port 8000) │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Redis     │    │ TimescaleDB │
                   │ (Port 6379) │    │ (Port 5432) │
                   └─────────────┘    └─────────────┘
```

## 🛠️ Configuration

### Frontend Configuration
- **Framework**: Next.js 15 with TypeScript
- **Styling**: Tailwind CSS with dark theme
- **Real-time**: Socket.IO client
- **Charts**: Recharts for data visualization

### Backend Configuration
- **Framework**: FastAPI with WebSocket support
- **Database**: TimescaleDB for time-series data
- **Caching**: Redis for session and real-time data
- **Integration**: Direct connection to moneytrailz system

### Database Schema
The system automatically creates the following tables:
- `portfolio_snapshots` - Real-time portfolio data
- `strategy_updates` - Strategy performance tracking
- `trades` - Trade execution history
- `positions` - Current position tracking

## 📊 Monitoring & Alerts

### Grafana Dashboards
Access Grafana at `https://your-domain.com:3001`

Default dashboards include:
- Portfolio Performance Overview
- Strategy Monitoring
- System Health Metrics
- Real-time Trading Activity

### Prometheus Metrics
Key metrics monitored:
- Portfolio value changes
- Strategy P&L tracking
- API response times
- WebSocket connection status
- Database performance

### Alert Configuration
Configure alerts in `monitoring/prometheus.yml`:

```yaml
# Portfolio Loss Alert
- alert: PortfolioLoss
  expr: portfolio_value_change < -5.0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Portfolio dropped by {{ $value }}%"

# Strategy Error Alert
- alert: StrategyError
  expr: strategy_errors > 3
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Strategy {{ $labels.strategy }} has {{ $value }} errors"
```

## 🔒 Security

### SSL/TLS Configuration
The system supports:
- TLS 1.2+ only
- HSTS headers
- Secure cookie settings
- CORS protection

### Authentication
- JWT-based authentication
- Session timeout (configurable)
- Rate limiting on API endpoints
- CSRF protection

### Data Protection
- Environment variable encryption
- Database connection encryption
- Redis AUTH configuration
- Log sanitization

## 🚦 Health Checks

### Service Health Endpoints
- Frontend: `GET /api/health`
- Backend: `GET /health`
- Database: Built-in PostgreSQL health check
- Redis: Built-in Redis ping check

### Monitoring Health
```bash
# Check all services
./scripts/health-check.sh

# Check specific service
curl https://your-domain.com/api/health
```

## 🔄 Backup & Recovery

### Automated Backups
Daily backups configured in `docker-compose.production.yml`:
```yaml
backup:
  image: postgres:14
  command: |
    bash -c '
    while true; do
      pg_dump -h database -U moneytrailz thetagang_dashboard > /backups/backup_$$(date +%Y%m%d_%H%M%S).sql
      find /backups -name "*.sql" -mtime +30 -delete
      sleep 86400
    done'
  volumes:
    - ./backups:/backups
```

### Manual Backup
```bash
# Create backup
docker exec moneytrailz-database pg_dump -U moneytrailz thetagang_dashboard > backup.sql

# Restore backup
docker exec -i moneytrailz-database psql -U moneytrailz thetagang_dashboard < backup.sql
```

## 🐛 Troubleshooting

### Common Issues

**1. Frontend not loading**
```bash
# Check frontend logs
docker-compose -f docker-compose.production.yml logs frontend

# Restart frontend
docker-compose -f docker-compose.production.yml restart frontend
```

**2. Backend API errors**
```bash
# Check backend logs
docker-compose -f docker-compose.production.yml logs backend

# Check database connection
docker-compose -f docker-compose.production.yml exec backend python -c "import asyncpg; print('OK')"
```

**3. WebSocket connection issues**
```bash
# Check WebSocket endpoint
wscat -c wss://your-domain.com/socket.io/?transport=websocket

# Check CORS settings
curl -H "Origin: https://your-domain.com" https://your-domain.com/api/health
```

**4. Database connection errors**
```bash
# Check database status
docker-compose -f docker-compose.production.yml exec database pg_isready -U moneytrailz

# Reset database
docker-compose -f docker-compose.production.yml down -v
docker-compose -f docker-compose.production.yml up -d database
```

### Performance Tuning

**Database Optimization**
```sql
-- Create indexes for better performance
CREATE INDEX idx_portfolio_timestamp ON portfolio_snapshots(timestamp);
CREATE INDEX idx_strategy_name ON strategy_updates(strategy_name);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
```

**Redis Configuration**
```bash
# Optimize Redis memory usage
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET maxmemory 256mb
```

**Nginx Optimization**
```nginx
# Enable gzip compression
gzip on;
gzip_types text/plain application/json application/javascript text/css;

# Enable caching
location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## 📝 Maintenance

### Regular Tasks
- [ ] Update SSL certificates (every 90 days for Let's Encrypt)
- [ ] Review and rotate API keys monthly
- [ ] Check backup integrity weekly
- [ ] Monitor disk space usage
- [ ] Review application logs

### Updates
```bash
# Update to latest version
git pull origin main
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d
```

### Scaling
For high-traffic deployments:
```yaml
# Scale frontend
docker-compose -f docker-compose.production.yml up -d --scale frontend=3

# Scale backend
docker-compose -f docker-compose.production.yml up -d --scale backend=2
```

## 📞 Support

For technical support:
- Check logs: `docker-compose -f docker-compose.production.yml logs`
- Review metrics: Access Grafana dashboard
- Submit issues: GitHub Issues
- Documentation: [moneytrailz Dashboard Wiki](link-to-wiki)

## 🚀 Advanced Configuration

### Custom Domain Setup
1. Point your domain's A record to your server IP
2. Update `NEXT_PUBLIC_API_URL` in environment
3. Configure SSL certificates
4. Update CORS origins

### Integration with moneytrailz
The dashboard integrates with your existing moneytrailz setup:
- Mount moneytrailz directory: `/moneytrailz-system`
- Configure data path: `THETAGANG_DATA_PATH`
- Set config path: `THETAGANG_CONFIG_PATH`

### Custom Strategies
Add custom strategy monitoring:
1. Implement strategy in moneytrailz
2. Update dashboard backend integration
3. Add strategy cards to frontend
4. Configure alerts in Grafana

---

**🎯 Ready to deploy your moneytrailz Dashboard!** 

This production setup provides a robust, scalable, and secure environment for monitoring your algorithmic trading performance. 

# Luminaria Book Recommender - Performance Optimizations

This document outlines the performance optimizations implemented in the Luminaria Book Recommender application to improve response times, reduce API calls, and enhance user experience.

## Implemented Optimizations

### 1. Multi-level Caching System
- **In-memory TTL Caching**: Added time-based caching for search results, book information, and news updates
- **Normalized Cache Keys**: Implemented consistent key generation for reliable cache hits
- **Force Refresh Option**: Added ability to bypass cache when fresh data is required

### 2. Batch Processing System
- **Batched API Calls**: Process multiple book recommendations in batches to reduce API call volume
- **Parallel Data Enrichment**: Use ThreadPoolExecutor to fetch additional book details concurrently
- **Improved Error Handling**: Better error recovery to ensure partial results are still returned

### 3. Asynchronous Loading
- **Background Jobs System**: Added a complete background job processing system for expensive operations
- **Job Status Tracking**: API endpoints to check status of running background jobs
- **Automatic Cleanup**: Periodic cleanup of completed jobs to prevent memory leaks

### 4. Cache Preloading
- **Popular Search Prefetching**: Automatically prefetch information for popular search terms
- **Scheduled Updates**: Daily cache refresh for frequently accessed content
- **Admin API**: Endpoint for manually refreshing specific cache items

### 5. Optimized UI Loading
- **Asynchronous News Updates**: Load news and social media updates after the main content
- **Non-blocking Content**: Ensure core book recommendations load quickly without waiting for secondary data

## API Endpoints

### Background Jobs
- `GET /api/job/<job_id>`: Check status of a background job

### Cache Management
- `POST /api/cache/refresh`: Refresh specific cache items
  - Required parameters:
    - `cache_type`: Type of cache to refresh (author, book, genre, news, recommendations, all)
    - `search_term`: The search term to refresh cache for

### Asynchronous Content
- `GET /api/fetch_updates?search_term=<term>`: Fetch news and social updates for a search term

## Memory Usage Considerations

The caching system has been designed with memory constraints in mind:
- TTL (Time To Live) limits on cache entries to prevent stale data accumulation
- Maxsize restrictions on caches to avoid unbounded growth
- Automatic cleanup of old background jobs and temporary resources

## Monitoring and Management

To monitor performance:
- Set `DEBUG=True` in environment to enable detailed logging
- Review logs for warnings about cache size limits being reached
- Use the cache refresh API endpoint to manually update stale data

## Further Optimizations (Future Work)

Potential future enhancements:
- Implement database-backed caching for persistence across app restarts
- Add a rate limiting system for external API calls
- Introduce cache prewarming on application startup
- Expand batch processing to other expensive operations

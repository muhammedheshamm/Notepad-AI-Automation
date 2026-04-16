import logging
from typing import Dict, List

import requests

log = logging.getLogger(__name__)

_POSTS_URL = "https://jsonplaceholder.typicode.com/posts"



def fetch_posts(limit: int = 10) -> List[Dict]:
    """Fetch the first `limit` posts from JSONPlaceholder, falling back to generated posts."""
    log.info("Fetching %d posts from %s", limit, _POSTS_URL)
    try:
        response = requests.get(_POSTS_URL, timeout=10)
        response.raise_for_status()
        posts = [p for p in response.json()[:limit] if validate_post(p)]
        log.info("Fetched %d posts from API", len(posts))
        return posts
    except Exception as exc:
        log.warning("API unavailable (%s) — using generated fallback posts", exc)
        return _generate_fallback_posts(limit)


def _generate_fallback_posts(limit: int) -> List[Dict]:
    """Generate simple numbered posts when the API is unreachable."""
    posts = [
        {
            "id": i,
            "userId": 1,
            "title": f"Post number {i}",
            "body": (
                f"This is the body of post number {i}.\n"
                "Generated as fallback because the API was unavailable."
            ),
        }
        for i in range(1, limit + 1)
    ]
    log.info("Generated %d fallback posts", len(posts))
    return posts


def validate_post(post: Dict) -> bool:
    """Return True if the post has all required fields."""
    return all(field in post for field in ["id", "title", "body"])


def format_post_content(post: Dict) -> str:
    """Format a post dict for writing into a text file."""
    return f"Title: {post.get('title', '')}\n\n{post.get('body', '')}"


def post_filename(post: Dict) -> str:
    """Return the output filename for a post."""
    return f"post_{post['id']}.txt"

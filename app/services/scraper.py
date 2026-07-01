import asyncio
import re
from typing import List

import httpx
from bs4 import BeautifulSoup


def _normalize_url(url: str) -> str:
    """Add https:// scheme if missing (handles 'github.com/user' style inputs)."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


async def _fetch_github(url: str) -> str:
    """Fetches profile bio, repo list, and language spread via the GitHub public API."""
    username = url.rstrip("/").split("github.com/")[-1].split("/")[0].split("?")[0]
    if not username:
        return f"[GitHub] Could not parse username from: {url}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "CVScreener/1.0",
    }
    user_api = f"https://api.github.com/users/{username}"
    repos_api = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=8"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            user_resp, repos_resp = await asyncio.gather(
                client.get(user_api, headers=headers),
                client.get(repos_api, headers=headers),
            )

            lines = [f"=== GitHub Profile ({url}) ==="]

            if user_resp.status_code == 200:
                u = user_resp.json()
                if u.get("name"):
                    lines.append(f"Name: {u['name']}")
                if u.get("bio"):
                    lines.append(f"Bio: {u['bio']}")
                if u.get("company"):
                    lines.append(f"Company: {u['company']}")
                lines.append(
                    f"Public repos: {u.get('public_repos', 'N/A')}  |  "
                    f"Followers: {u.get('followers', 0)}"
                )

            if repos_resp.status_code == 200:
                repos = repos_resp.json()
                if repos:
                    lines.append("\nRecent Repositories:")
                    for r in repos[:8]:
                        lang = r.get("language") or "N/A"
                        desc = (r.get("description") or "")[:120]
                        stars = r.get("stargazers_count", 0)
                        lines.append(f"  - {r['name']} [{lang}] ★{stars}: {desc}")

                    languages = sorted(
                        {r.get("language") for r in repos if r.get("language")}
                    )
                    if languages:
                        lines.append(f"\nLanguages across repos: {', '.join(languages)}")

            return "\n".join(lines)

        except Exception as exc:
            return f"[GitHub] Error fetching {url}: {exc}"


async def _fetch_portfolio(url: str) -> str:
    """
    Scrapes clean readable text from a personal portfolio or website.
    Also extracts GitHub/LinkedIn profile links that are hidden behind icons
    (e.g. <a href="https://github.com/user"><img .../></a>) and fetches
    GitHub API data for any discovered profiles.
    """
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        try:
            response = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    )
                },
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # --- Extract social links from <a href> BEFORE removing tags ---
                # Catches icon-only links where visible text is absent
                found_github: list[str] = []
                found_linkedin: list[str] = []
                for a in soup.find_all("a", href=True):
                    href = _normalize_url(a["href"].strip())
                    if "github.com/" in href:
                        # Only user-profile URLs (one path segment after github.com/)
                        after = href.rstrip("/").split("github.com/")[-1]
                        if after and "/" not in after and href not in found_github:
                            found_github.append(href)
                    elif "linkedin.com/in/" in href and href not in found_linkedin:
                        found_linkedin.append(href)

                # Remove boilerplate before extracting visible text
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()

                lines = [f"=== Portfolio/Website ({url}) ===", text[:3500]]

                # Report any icon-linked social profiles found
                if found_github or found_linkedin:
                    lines.append("\nSocial Profile Links Detected (icon/button links):")
                    for gh in found_github[:3]:
                        lines.append(f"  GitHub: {gh}")
                    for li in found_linkedin[:3]:
                        lines.append(f"  LinkedIn: {li}")

                # Also fetch full GitHub API data for discovered profiles
                if found_github:
                    github_results = await asyncio.gather(
                        *[_fetch_github(gh) for gh in found_github[:2]]
                    )
                    lines.extend(github_results)

                return "\n".join(lines)
            return f"[Portfolio] HTTP {response.status_code} for: {url}"

        except Exception as exc:
            return f"[Portfolio] Error fetching {url}: {exc}"


async def _linkedin_note(url: str) -> str:
    """LinkedIn blocks scraping; return a minimal note so the LLM knows the URL exists."""
    return (
        f"=== LinkedIn Profile ===\n"
        f"URL: {url}\n"
        f"Note: LinkedIn data is not accessible via automated scraping. "
        f"The candidate maintains a LinkedIn presence at the URL above."
    )


async def scrape_links(links: List[str]) -> str:
    """
    Scrapes each profile/portfolio link concurrently.
    Returns combined text ready to be appended to the LLM prompt as supplementary context.

    Supported link types:
      - github.com   → GitHub public API (repos, bio, languages)
      - linkedin.com → URL-only note (scraping blocked)
      - anything else → HTTP + BeautifulSoup full-page text
    """
    if not links:
        return ""

    tasks = []
    for raw in links:
        link = _normalize_url(raw)  # handles missing https:// scheme
        if not link:
            continue
        if "github.com" in link:
            tasks.append(_fetch_github(link))
        elif "linkedin.com" in link:
            tasks.append(_linkedin_note(link))
        else:
            tasks.append(_fetch_portfolio(link))

    if not tasks:
        return ""

    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [str(r) for r in results if not isinstance(r, Exception)]
    return "\n\n---\n\n".join(valid) if valid else ""

# scrape_seleniumbase.py - Email Scraper using SeleniumBase
"""
Email scraper using SeleniumBase for better stealth and reliability.
Compatible with Python 3.7+

Installation:
pip install seleniumbase beautifulsoup4 requests dnspython
"""

import re
import socket
import time
import random
import csv
import threading
from urllib.parse import urlparse, urljoin, quote_plus
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from seleniumbase import Driver
from selenium.webdriver.common.keys import Keys
import requests

# Optional: MX validation
try:
    import dns.resolver
    HAS_DNSPY = True
except Exception:
    HAS_DNSPY = False

# -------------------- CONFIG --------------------
MAX_RESULTS_PER_QUERY = 10      # Results per search query
THREADS = 1                      # Single thread for more human-like behavior
PAGE_LOAD_TIMEOUT = 30
DELAY_MIN = 4.0                  # Minimum delay between actions (seconds)
DELAY_MAX = 8.0                  # Maximum delay between actions (seconds)
MX_CHECK = True                  # Validate email domains
IPINFO_TOKEN = None
HEADLESS = False                 # Set True to hide browser (not recommended for stealth)

OUTPUT_CSV = "usa_emails_selenium.csv"
LOCK = threading.Lock()
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Store one driver per thread
_drivers = {}
# ------------------------------------------------

def get_driver():
    """Get or create Chrome driver using SeleniumBase."""
    thread_id = threading.get_ident()
    
    if thread_id not in _drivers:
        print("    [DEBUG] Creating new SeleniumBase driver...")
        
        try:
            # SeleniumBase Driver with stealth mode
            driver = Driver(
                browser="chrome",
                uc=True,  # Use undetected mode
                headless=HEADLESS,
                incognito=False,
                agent=random_user_agent(),  # Random user agent
                do_not_track=True,
                undetectable=True,
                page_load_strategy="eager"  # Changed to eager - don't wait for everything
            )
            
            # Set timeouts
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            driver.implicitly_wait(10)
            
            # Randomize window size (human-like)
            widths = [1366, 1440, 1536, 1920]
            heights = [768, 900, 864, 1080]
            width = random.choice(widths)
            height = random.choice(heights)
            driver.set_window_size(width, height)
            
            # Random initial position
            if not HEADLESS:
                try:
                    driver.set_window_position(random.randint(0, 100), random.randint(0, 100))
                except:
                    pass
            
            # Add some browser entropy
            driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Randomize plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Randomize languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
            """)
            
            _drivers[thread_id] = driver
            print("    [DEBUG] SeleniumBase driver created successfully!")
            return driver
            
        except Exception as e:
            print("    [ERROR] Failed to create driver: {}".format(e))
            print("    [HELP] Make sure Chrome browser is installed")
            print("    [HELP] Try running: pip install --upgrade seleniumbase")
            return None
    
    return _drivers[thread_id]

def close_driver():
    """Close driver for current thread."""
    thread_id = threading.get_ident()
    if thread_id in _drivers:
        try:
            _drivers[thread_id].quit()
            del _drivers[thread_id]
        except:
            pass

def wait_random():
    """Human-like delay."""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def short_delay():
    """Short random delay like human thinking."""
    time.sleep(random.uniform(0.5, 1.5))

def micro_delay():
    """Micro delay between actions."""
    time.sleep(random.uniform(0.1, 0.3))

def scroll_page(driver):
    """Scroll page slowly like a human reading."""
    try:
        # Random number of scrolls
        num_scrolls = random.randint(2, 5)
        
        for i in range(num_scrolls):
            # Variable scroll distances
            scroll_amount = random.randint(150, 500)
            driver.execute_script("window.scrollBy(0, {});".format(scroll_amount))
            
            # Random pause like human reading
            time.sleep(random.uniform(0.3, 1.2))
            
            # Sometimes scroll back up a bit (like re-reading)
            if random.random() < 0.3:
                driver.execute_script("window.scrollBy(0, {});".format(random.randint(-100, -50)))
                time.sleep(random.uniform(0.2, 0.5))
    except:
        pass

def move_mouse_randomly(driver):
    """Simulate random mouse movements."""
    try:
        # Move mouse to random positions
        for _ in range(random.randint(1, 3)):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            driver.execute_script("""
                var event = new MouseEvent('mousemove', {{
                    'view': window,
                    'bubbles': true,
                    'cancelable': true,
                    'clientX': {},
                    'clientY': {}
                }});
                document.dispatchEvent(event);
            """.format(x, y))
            time.sleep(random.uniform(0.1, 0.3))
    except:
        pass

def human_type(element, text, driver):
    """Type text like a human with realistic delays."""
    try:
        element.clear()
        micro_delay()
        
        for char in text:
            element.send_keys(char)
            # Variable typing speed
            delay = random.uniform(0.05, 0.15)
            # Sometimes pause longer (thinking)
            if random.random() < 0.1:
                delay = random.uniform(0.3, 0.7)
            time.sleep(delay)
        
        # Small pause after typing
        time.sleep(random.uniform(0.3, 0.8))
    except:
        pass

def random_user_agent():
    """Get a random user agent."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    return random.choice(user_agents)

# ---------- Search using Selenium ----------

def search_google(query, max_results=10):
    """Search Google using human-like interactions."""
    driver = get_driver()
    if not driver:
        return []
    
    urls = []
    
    try:
        print("    [DEBUG] Loading Google homepage...")
        
        # Go to Google homepage first (more human-like)
        driver.execute_script("window.location.href = 'https://www.google.com';")
        short_delay()
        
        # Stop loading if needed
        try:
            driver.execute_script("window.stop();")
        except:
            pass
        
        time.sleep(random.uniform(1, 2))
        
        # Random mouse movements
        move_mouse_randomly(driver)
        
        # Accept cookies if needed
        try:
            buttons = driver.find_elements("xpath", "//button")
            for btn in buttons[:5]:
                try:
                    text = btn.text.lower()
                    if 'accept' in text or 'agree' in text or 'ok' in text:
                        print("    [DEBUG] Accepting cookies...")
                        # Move mouse to button area first
                        driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        micro_delay()
                        btn.click()
                        short_delay()
                        break
                except:
                    continue
        except:
            pass
        
        # Find search box and type query like human
        try:
            print("    [DEBUG] Typing search query: {}".format(query))
            
            # Try multiple selectors for search box
            search_box = None
            selectors = [
                "textarea[name='q']",
                "input[name='q']",
                "textarea[title*='Search']",
                "input[title*='Search']"
            ]
            
            for selector in selectors:
                try:
                    search_box = driver.find_element("css selector", selector)
                    if search_box:
                        break
                except:
                    continue
            
            if not search_box:
                print("    [ERROR] Could not find search box")
                return []
            
            # Click on search box first
            driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
            micro_delay()
            search_box.click()
            short_delay()
            
            # Type query like human
            human_type(search_box, query, driver)
            
            # Press Enter
            search_box.send_keys(Keys.RETURN)
            
            print("    [DEBUG] Waiting for search results...")
            
        except Exception as e:
            print("    [ERROR] Failed to type in search box: {}".format(e))
            return []
        
        # Wait for results page to load
        start_time = time.time()
        while time.time() - start_time < 8:
            try:
                if driver.execute_script("return document.readyState") in ["complete", "interactive"]:
                    # Check if we have results
                    if "google.com/search" in driver.current_url:
                        break
            except:
                pass
            time.sleep(0.5)
        
        # Stop loading
        try:
            driver.execute_script("window.stop();")
        except:
            pass
        
        time.sleep(random.uniform(1.5, 2.5))
        
        # Random mouse movements and scrolling (like reading results)
        move_mouse_randomly(driver)
        scroll_page(driver)
        
        # Extract search results
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Try multiple selectors for Google results
        for a in soup.select("a"):
            href = a.get("href", "")
            
            # Clean Google redirect URLs
            if '/url?q=' in href:
                try:
                    href = href.split('/url?q=')[1].split('&')[0]
                except:
                    continue
            
            # Filter valid URLs
            if href.startswith("http") and "google.com" not in href and "youtube.com" not in href:
                if href not in urls:
                    urls.append(href)
                if len(urls) >= max_results:
                    break
        
        print("    [DEBUG] Found {} URLs from Google".format(len(urls)))
        
    except Exception as e:
        print("    [ERROR] Google search failed: {}".format(e))
        import traceback
        print("    [TRACE] {}".format(traceback.format_exc()[:200]))
    
    return urls[:max_results]

def search_duckduckgo(query, max_results=10):
    """Search DuckDuckGo using SeleniumBase."""
    driver = get_driver()
    if not driver:
        return []
    
    urls = []
    search_url = "https://duckduckgo.com/?q={}".format(quote_plus(query))
    
    try:
        print("    [DEBUG] Loading DuckDuckGo search...")
        
        # Use JavaScript to navigate
        driver.execute_script("window.location.href = arguments[0];", search_url)
        
        # Wait but stop if taking too long
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                if driver.execute_script("return document.readyState") in ["complete", "interactive"]:
                    break
            except:
                pass
            time.sleep(0.3)
        
        try:
            driver.execute_script("window.stop();")
        except:
            pass
        
        time.sleep(2)
        
        scroll_page(driver)
        time.sleep(1)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        for a in soup.select("a"):
            href = a.get("href", "")
            if href.startswith("http") and "duckduckgo.com" not in href:
                if href not in urls:
                    urls.append(href)
                if len(urls) >= max_results:
                    break
        
        print("    [DEBUG] Found {} URLs from DuckDuckGo".format(len(urls)))
        
    except Exception as e:
        print("    [ERROR] DuckDuckGo search failed: {}".format(e))
    
    return urls[:max_results]

def discover_websites(query, max_results=10):
    """Search for websites."""
    print("\n[+] Searching for: {}".format(query))
    
    # Try Google first
    urls = search_google(query, max_results)
    
    # If not enough, try DuckDuckGo
    if len(urls) < max_results // 2:
        print("    [DEBUG] Trying DuckDuckGo as backup...")
        ddg_urls = search_duckduckgo(query, max_results)
        urls.extend(ddg_urls)
    
    # Remove duplicates
    unique = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    
    # Count unique domains
    domains = set()
    for url in unique[:max_results]:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            if domain:
                domains.add(domain)
        except:
            pass
    
    print("    [RESULT] Found {} URLs ({} unique domains)".format(len(unique[:max_results]), len(domains)))
    
    return unique[:max_results]

# ---------- Email extraction ----------

def extract_emails_from_text(text):
    """Extract email addresses from text."""
    emails = set(re.findall(EMAIL_REGEX, text))
    
    # Filter out garbage
    filtered = set()
    for email in emails:
        email_lower = email.lower()
        # Skip image files, etc.
        if email_lower.endswith(('.png', '.jpg', '.gif', '.css', '.js')):
            continue
        # Skip example domains
        if any(x in email_lower for x in ['example.com', 'test.com', 'localhost']):
            continue
        filtered.add(email)
    
    return filtered

def get_ip_country(domain):
    """Get country code for domain."""
    try:
        ip = socket.gethostbyname(domain)
        r = requests.get("https://ipinfo.io/{}/json".format(ip), timeout=5)
        time.sleep(0.3)
        data = r.json()
        return data.get("country")
    except:
        return None

def is_usa_website(url):
    """Check if website is USA-based."""
    try:
        netloc = urlparse(url).netloc.replace("www.", "")
        country = get_ip_country(netloc)
        return country == "US"
    except:
        return False

def mx_check(domain):
    """Check if domain has valid MX records."""
    if not HAS_DNSPY or not MX_CHECK:
        return True
    try:
        dns.resolver.resolve(domain, 'MX')
        return True
    except:
        try:
            dns.resolver.resolve(domain, 'A')
            return True
        except:
            return False

def get_contact_pages(base_url):
    """Generate list of potential contact page URLs."""
    parsed = urlparse(base_url)
    base = "{}://{}".format(parsed.scheme, parsed.netloc)
    
    pages = [
        base + "/",
        base + "/contact",
        base + "/contact-us",
        base + "/about",
        base + "/about-us",
    ]
    return pages

def scrape_emails_from_site(base_url):
    """Scrape emails from a website using human-like behavior."""
    driver = get_driver()
    if not driver:
        return set()
    
    emails_found = set()
    
    # Get the base domain to build contact pages
    parsed = urlparse(base_url)
    base_domain = "{}://{}".format(parsed.scheme if parsed.scheme else "https", parsed.netloc)
    
    # Generate contact page URLs from the base domain (not the specific URL)
    pages = get_contact_pages(base_domain)
    
    for page in pages[:3]:  # Only check first 3 pages
        try:
            print("    [DEBUG] Visiting: {}".format(page))
            
            # Human-like delay before visiting next page
            time.sleep(random.uniform(2, 4))
            
            # Try loading with timeout protection
            try:
                # Start loading page
                driver.execute_script("window.location.href = arguments[0];", page)
                
                # Wait but give up if it takes too long
                start_time = time.time()
                while time.time() - start_time < 6:  # Slightly longer for content pages
                    try:
                        # Check if we can access the page
                        ready_state = driver.execute_script("return document.readyState")
                        if ready_state in ["complete", "interactive"]:
                            break
                    except:
                        pass
                    time.sleep(0.5)
                
                # Stop loading to prevent timeout
                try:
                    driver.execute_script("window.stop();")
                except:
                    pass
                
                time.sleep(random.uniform(1, 2))
                
            except Exception as load_error:
                print("    [DEBUG] Load issue: {}, trying to continue...".format(str(load_error)[:50]))
                try:
                    driver.execute_script("window.stop();")
                except:
                    pass
                time.sleep(1)
            
            # Human-like behavior: move mouse and scroll
            move_mouse_randomly(driver)
            short_delay()
            
            # Scroll through page like reading
            scroll_page(driver)
            
            # Another small delay
            time.sleep(random.uniform(0.5, 1.5))
            
            # Try to get page source
            try:
                html = driver.page_source
                
                # Check if we got any content
                if not html or len(html) < 100:
                    print("    [DEBUG] No content retrieved, skipping...")
                    continue
                    
            except Exception as e:
                print("    [DEBUG] Cannot get page source: {}, skipping...".format(str(e)[:50]))
                continue
            
            # Extract emails from HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Check mailto links first (most reliable)
            for link in soup.select("a[href^='mailto:']"):
                try:
                    email = link['href'].replace('mailto:', '').split('?')[0].strip()
                    if email:
                        emails_found.add(email)
                except:
                    pass
            
            # Extract from text
            emails_found.update(extract_emails_from_text(html))
            
        except Exception as e:
            print("    [DEBUG] Error on {}: {}".format(page, str(e)[:100]))
            continue
    
    # Validate emails
    valid_emails = set()
    for email in emails_found:
        try:
            domain = email.split('@')[1]
            if mx_check(domain):
                valid_emails.add(email.lower())
        except:
            continue
    
    return valid_emails

# ---------- CSV output ----------

def write_emails_to_csv(rows):
    """Save emails to CSV file."""
    with LOCK:
        try:
            with open(OUTPUT_CSV, "r"):
                write_header = False
        except:
            write_header = True
        
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["email"])
            for row in rows:
                writer.writerow(row)

# ---------- Main processing ----------

def process_site(site, scraped_domains):
    """Process a single website."""
    try:
        parsed = urlparse(site)
        domain = parsed.netloc.replace("www.", "")
        
        if not domain:
            return
        
        # Check if we already scraped this domain
        if domain in scraped_domains:
            print("\n[*] Skipping: {} (already scraped)".format(site))
            return
        
        if not parsed.scheme:
            site = "http://" + site
        
        print("\n[*] Processing: {}".format(site))
        
        # Mark domain as scraped
        scraped_domains.add(domain)
        
        # Check if USA
        if not is_usa_website(site):
            print("    [SKIP] Not US-hosted: {}".format(domain))
            return
        
        print("    [OK] US-hosted, scraping emails...")
        
        # Scrape emails
        emails = scrape_emails_from_site(site)
        
        if not emails:
            print("    [RESULT] No emails found")
            return
        
        # Save to CSV
        rows = [(email, site, domain, "US") for email in emails]
        write_emails_to_csv(rows)
        
        print("    [RESULT] Saved {} emails!".format(len(rows)))
        
    except Exception as e:
        print("    [ERROR] Failed: {}".format(e))

# ---------- Main ----------

def main():
    print("=" * 70)
    print("USA Email Scraper - SeleniumBase Version")
    print("=" * 70)
    print()
    
    # Your search queries
    queries = [
        "doctor New York contact email",
        "lawyer California email",
    ]
    
    print("[+] Step 1: Discovering websites...")
    print()
    
    all_sites = []
    for query in queries:
        sites = discover_websites(query, max_results=MAX_RESULTS_PER_QUERY)
        all_sites.extend(sites)
        wait_random()
    
    # Close search driver
    close_driver()
    time.sleep(2)
    
    # Remove duplicates and group by domain
    print("\n[+] Filtering duplicate domains...")
    unique_sites = []
    seen_domains = set()
    
    for site in all_sites:
        try:
            parsed = urlparse(site)
            domain = parsed.netloc.replace("www.", "")
            
            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                unique_sites.append(site)
        except:
            continue
    
    print("[+] Found {} unique domains (filtered {} duplicates)".format(
        len(unique_sites), len(all_sites) - len(unique_sites)))
    print()
    print("[+] Step 2: Scraping emails...")
    print()
    
    # Track domains we've already scraped (in case of any remaining duplicates)
    scraped_domains = set()
    
    # Process sites
    for site in unique_sites:
        process_site(site, scraped_domains)
        time.sleep(random.uniform(2, 4))  # Delay between sites
    
    # Cleanup
    print("\n[+] Cleaning up...")
    for driver in list(_drivers.values()):
        try:
            driver.quit()
        except:
            pass
    
    print("\n" + "=" * 70)
    print("DONE! Check {}".format(OUTPUT_CSV))
    print("=" * 70)

if __name__ == "__main__":
    if MX_CHECK and not HAS_DNSPY:
        print("Warning: dnspython not installed, MX validation disabled")
        print("Install with: pip install dnspython")
        print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user. Cleaning up...")
        for driver in list(_drivers.values()):
            try:
                driver.quit()
            except:
                pass
    except Exception as e:
        print("\nError: {}".format(e))
        import traceback
        traceback.print_exc()
        for driver in list(_drivers.values()):
            try:
                driver.quit()
            except:
                pass
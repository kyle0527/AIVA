import globals
import json
import csv
from collections import defaultdict

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = DummyColor()
    Style = DummyColor()


def print_report(start_time, end_time):
    print(f"\n=== {Fore.CYAN}Layer 7 Test Completed{Style.RESET_ALL} ===")
    duration = end_time - start_time
    print(f"Total time: {Fore.YELLOW}{duration:.2f}{Style.RESET_ALL} seconds")
    print(f"Successful requests: {Fore.GREEN}{globals.results['success']}{Style.RESET_ALL}")
    print(f"Failed requests: {Fore.RED}{globals.results['fail']}{Style.RESET_ALL}")
    
    total_reqs = globals.results['success'] + globals.results['fail']
    if total_reqs > 0 and duration > 0:
        rps = total_reqs / duration
        print(f"Requests Per Second (RPS): {Fore.YELLOW}{rps:.2f}{Style.RESET_ALL}")

    if globals.results["details"]:
        print(f"\n--- {Fore.CYAN}Response Details{Style.RESET_ALL} ---")
        sorted_details = sorted(globals.results["details"].items(), key=lambda item: str(item[0]))
        for code, count in sorted_details:
            color = Fore.GREEN if isinstance(code, int) and 200 <= code < 400 else Fore.RED
            print(f"  - {color}{code}{Style.RESET_ALL}: {count} responses")


def print_flood_report(start_time, end_time):
    print(f"\n=== {Fore.CYAN}Layer 3/4 Flood Completed{Style.RESET_ALL} ===")
    duration = end_time - start_time
    print(f"Total time: {Fore.YELLOW}{duration:.2f}{Style.RESET_ALL} seconds")
    
    packets = globals.results['packets_sent']
    total_bytes = globals.results['bytes_sent']
    
    print(f"Total Packets Sent: {Fore.GREEN}{packets:,}{Style.RESET_ALL}")
    
    if duration > 0:
        pps = packets / duration
        print(f"Packets Per Second (PPS): {Fore.YELLOW}{pps:,.2f}{Style.RESET_ALL}")

    if total_bytes > 0:
        total_mbit = (total_bytes * 8) / (1024 * 1024)
        print(f"Total Data Sent: {Fore.GREEN}{total_mbit:.2f}{Style.RESET_ALL} Mbit")
        if duration > 0:
            mbps = total_mbit / duration
            print(f"Average Bandwidth: {Fore.YELLOW}{mbps:.2f}{Style.RESET_ALL} Mbps")



def print_dns_check_report():
    all_ips = globals.resolved_ip_details.values()
    valid_ips = [details for details in all_ips if details.get('status') != 'failed' and details.get('status') != 'pending']
    failed_count = len(list(all_ips)) - len(valid_ips)

    if not valid_ips and failed_count == 0:
        print("\nNo detailed DNS records found for the given hostname.")
        return

    print("\n--- DNS Resolution Report ---")
    print(f"\n[+] Summary: Found {len(set(d['query'] for d in valid_ips))} unique IP(s) with details. {failed_count} lookup(s) failed.")

    country_distribution = defaultdict(int)
    for details in valid_ips:
        country_distribution[details.get('country', 'Unknown')] += 1
    
    print("\n[+] Geographical Distribution:")
    for country, count in sorted(country_distribution.items()):
        print(f"  - {country}: {count} IP(s)")

    grouped_by_asn = defaultdict(list)
    for details in valid_ips:
        asn = details.get('as', 'Unknown ASN')
        grouped_by_asn[asn].append(details)

    for asn, ip_list in sorted(grouped_by_asn.items()):
        print(f"\n[+] Network: {asn} ({len(ip_list)} IPs found)")
        
        IP_WIDTH, TYPE_WIDTH, COUNTRY_WIDTH, CITY_WIDTH, ISP_WIDTH = 40, 6, 15, 20, 45
        
        print("-" * (IP_WIDTH + TYPE_WIDTH + COUNTRY_WIDTH + CITY_WIDTH + ISP_WIDTH + 4))
        print(f"{'IP Address':<{IP_WIDTH}} {'Type':<{TYPE_WIDTH}} {'Country':<{COUNTRY_WIDTH}} {'City':<{CITY_WIDTH}} {'ISP':<{ISP_WIDTH}}")
        print(f"{'-'*IP_WIDTH:<{IP_WIDTH}} {'-'*TYPE_WIDTH:<{TYPE_WIDTH}} {'-'*COUNTRY_WIDTH:<{COUNTRY_WIDTH}} {'-'*CITY_WIDTH:<{CITY_WIDTH}} {'-'*ISP_WIDTH:<{ISP_WIDTH}}")
        
        for details in sorted(ip_list, key=lambda x: x.get('query')):
            ip = details.get('query', 'N/A')
            ip_type = details.get('type', 'N/A')
            country = details.get('country', 'N/A')
            city = details.get('city', 'N/A')
            isp = details.get('isp', 'N/A')
            
            print(f"{ip:<{IP_WIDTH}} {ip_type:<{TYPE_WIDTH}} {country:<{COUNTRY_WIDTH}} {city:<{CITY_WIDTH}} {isp:<{ISP_WIDTH}}")

def export_dns_report(filename):
    valid_ips = [details for details in globals.resolved_ip_details.values() if details.get('status') != 'failed' and details.get('status') != 'pending']
    if not valid_ips:
        print(f"\nNo data to export to {filename}.")
        return

    try:
        if filename.lower().endswith('.json'):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(valid_ips, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully exported report to {filename}")
        elif filename.lower().endswith('.csv'):
            if not valid_ips: return
            headers = sorted(list(set(key for ip_data in valid_ips for key in ip_data.keys())))
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(valid_ips)
            print(f"\nSuccessfully exported report to {filename}")
        else:
            print(f"\nError: Unsupported file format '{filename}'. Please use .json or .csv.")
    except IOError as e:
        print(f"\nError writing to file {filename}: {e}")

def print_analysis_report(hostname, findings):
    print(f"\n=== Intelligence Analysis for: {hostname} ===")
    print("\n[+] DNS Analysis:")
    print(f"  - CNAME Record: {findings['cname'] or 'Not found.'}")
    if findings['ns_records']:
        print(f"  - Name Servers: {', '.join(findings['ns_records'])}")
    if findings['a_records']:
        print(f"  - IP Addresses (A Records): {', '.join(findings['a_records'])}")
        if len(findings['a_records']) > 1:
            print("  - Note: Multiple A records found, indicating load balancing or a CDN.")
    print("\n[+] HTTP Header Analysis:")
    if findings['headers']:
        server = findings['headers'].get('Server', 'Not Found')
        print(f"  - Server Header: {server}")
        interesting_headers = ['X-Cache', 'Via', 'X-Ar-Cache', 'CF-RAY', 'X-CDN', 'Age']
        found_headers = [f"{h}: {findings['headers'][h]}" for h in interesting_headers if h in findings['headers']]
        if found_headers:
            print(f"  - Other Relevant Headers: {', '.join(found_headers)}")
    else:
        print("  - Could not retrieve HTTP headers.")
    print("\n[+] IP & ASN Ownership Analysis:")
    if findings['ip_info']:
        ip = findings['ip_info'].get('query', 'N/A')
        asn = findings['ip_info'].get('as', 'N/A')
        org = findings['ip_info'].get('org', 'N/A')
        country = findings['ip_info'].get('country', 'N/A')
        print(f"  - Primary IP: {ip}")
        print(f"  - ASN (Network): {asn}")
        print(f"  - Organization: {org}")
        print(f"  - Country: {country}")
    else:
        print("  - Could not retrieve IP/ASN ownership information.")
    print("\n--- Conclusion ---")
    if findings['cdn_provider'] != 'Unknown':
        print(f"The target is likely behind a CDN: **{findings['cdn_provider']}**")
        print(f"  - Reason: {findings['detection_reason']}")
    else:
        print("The target does not appear to be behind a known, major CDN based on these tests.")
        print("  - It might be using a direct server, a less common CDN, or a custom proxy setup.")

def print_origin_report(hostname):
    results = globals.origin_results
    cdn_ips = results.get('cdn_ips', [])
    
    print(f"\n=== Origin Server Search Results for: {hostname} ===")
    
    if cdn_ips:
        print(f"\n[*] Known CDN IPs for {hostname}: {', '.join(cdn_ips)}")
        print("    (These IPs belong to the CDN and will be filtered from the results below)")

    all_found_ips = set()
    
    method_order = [
        'Certificate Transparency Scan', 
        'Passive DNS Analysis (DNSdumpster)', 
        'Email Record Analysis',
        'ArvanCloud Subdomain Scan', 
        'Cloudflare Subdomain Scan',
        'General Subdomain Scan'
    ]
    
    for method_name in method_order:
        if method_name in results.get('methods', {}):
            ips = results['methods'][method_name]
            if ips:
                print(f"\n[+] Potential Origin IPs found via '{method_name}':")
                for ip in sorted(ips):
                    highlight = ""
                    if 'Certificate' in method_name or 'DNSdumpster' in method_name or 'Email' in method_name:
                        highlight = " <--- LIKELY THE REAL SERVER!"
                    print(f"  - {ip}{highlight}")
                    all_found_ips.add(ip)
            
    if not all_found_ips:
        print("\n[-] No potential origin IPs were found using any of the available techniques.")
        print("    This indicates the target is well-configured and has no known IP history.")
        return
        
    print("\n--- Next Steps ---")
    print("You can now test these IPs directly to see if they host the target website.")
    print("Use the 'stress' command with the -Ip flag to target the IP directly.")
    print("\nExample:")
    example_ip = list(all_found_ips)[0]
    print(f"  python main.py stress -Ip {example_ip}:80 -t 10 -r 50 --attack-profile")



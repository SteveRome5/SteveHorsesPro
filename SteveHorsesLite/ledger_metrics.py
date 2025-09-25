import csv, sys, statistics, datetime
fn = sys.argv[1]
rows = [r for r in csv.DictReader(open(fn, newline='', encoding='utf-8'))]

bets = sum(float(r.get('BetAmount', 0) or 0) for r in rows)
rets = sum(float(r.get('Return', 0) or 0) for r in rows)
wins = sum(1 for r in rows if (r.get('Result', '').strip().upper() == 'WIN'))
edges = [float(r['EdgePct']) for r in rows if (r.get('EdgePct', '').strip() != '')]

roi = (rets - bets) / bets if bets > 0 else 0.0
print(f"Total: bets=${bets:,.0f} returns=${rets:,.0f}  ROI={roi*100:.1f}%  HitRate={(wins/len(rows)*100 if rows else 0):.1f}%  AvgEdge={(statistics.mean(edges) if edges else 0):.2f} pp")

def parse_date(r):
    try: return datetime.date.fromisoformat(r['Date'])
    except: return None

cut = datetime.date.today() - datetime.timedelta(days=14)
recent = [r for r in rows if (d := parse_date(r)) and d >= cut]
b = sum(float(r.get('BetAmount', 0) or 0) for r in recent)
ret = sum(float(r.get('Return', 0) or 0) for r in recent)
w = sum(1 for r in recent if (r.get('Result', '').strip().upper() == 'WIN'))
print(f"Last 14d: bets=${b:,.0f} returns=${ret:,.0f}  ROI={((ret-b)/b*100 if b else 0):.1f}%  HitRate={(w/len(recent)*100 if recent else 0):.1f}%")
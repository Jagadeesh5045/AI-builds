"""Generate a realistic synthetic dataset of customer support tickets.

Run: python src/data_gen.py [--n 2000] [--seed 42] [--out data/tickets.csv]

Output columns: id, text, category, priority
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

CATEGORIES = [
    "billing",
    "technical",
    "account",
    "delivery",
    "returns",
    "product_info",
    "complaint",
    "feedback",
]

# Templates: (template, category, priority). {slot} placeholders are filled per row.
_TEMPLATES = [
    # ---- billing ----
    ("Hi, I was charged {amount} twice on my {card} this month. My invoice number is {inv}. Can you refund the duplicate charge?", "billing", "high"),
    ("My subscription renewed at {amount} but I cancelled it {days} days ago. Please stop the charge and refund me.", "billing", "high"),
    ("Why is my bill {amount} this month when it is usually around {amount2}? Nothing has changed on my plan.", "billing", "medium"),
    ("The payment on my {card} failed but the funds are available. Can you retry the charge for invoice {inv}?", "billing", "medium"),
    ("I need a copy of invoice {inv} for my records. The billing portal link you sent is not working.", "billing", "low"),
    ("You charged me a late fee of {amount} but I paid on time — see payment confirmation {inv}. Please remove it.", "billing", "high"),
    ("My card {card} expires next month. How do I update my payment method before the next billing cycle?", "billing", "low"),
    ("I signed up for the annual plan but got billed monthly at {amount}. Please switch me over and adjust the charge.", "billing", "medium"),
    ("There is a charge of {amount} on my statement from {company} that I do not recognise. Can you investigate?", "billing", "high"),
    ("Please cancel my subscription effective immediately and confirm I will not be charged again.", "billing", "medium"),
    # ---- technical ----
    ("The app crashes every time I tap {feature}. I am on {os} version {ver}. This started after yesterday's update.", "technical", "high"),
    ("I keep getting error {code} when I try to export my report. It works fine for smaller files though.", "technical", "medium"),
    ("The dashboard will not load — I just see a spinning wheel. Tried on Chrome and Safari, same result.", "technical", "high"),
    ("Search returns no results even for terms I know exist. Cache cleared, still broken. {os} {ver}.", "technical", "medium"),
    ("Sync between my phone and desktop stopped working two days ago. Both show different data now.", "technical", "high"),
    ("The login page loops me back to itself after entering correct credentials. No error message at all.", "technical", "high"),
    ("Notifications stopped arriving on {os}. I checked settings, everything is enabled. Please help.", "technical", "medium"),
    ("The {feature} page shows outdated numbers — it has not refreshed since {day}. My colleague sees the same.", "technical", "medium"),
    ("Uploads fail at exactly 50% with error {code}. File is only 4MB, well under the limit.", "technical", "medium"),
    ("The whole site is down for me — {company}.com returns a 503. Is there an outage right now?", "technical", "urgent"),
    # ---- account ----
    ("I want to change the email on my account from {email} to {email2}. I no longer have access to the old one.", "account", "medium"),
    ("Someone logged into my account from another country. Please secure it and tell me what to do.", "account", "urgent"),
    ("I cannot reset my password — the reset link expires immediately every time. Tried three times now.", "account", "high"),
    ("Please delete my account and all my data. I have already exported what I needed.", "account", "medium"),
    ("I need to transfer ownership of our team workspace to my colleague {name}. How do we do that?", "account", "medium"),
    ("My account shows the wrong company name after our rebrand. Can you update it to {company}?", "account", "low"),
    ("Two-factor authentication codes are not arriving on my new phone. I am locked out of admin settings.", "account", "urgent"),
    ("I created two accounts by mistake ({email} and {email2}). Can you merge them and keep the first one?", "account", "low"),
    ("My free trial ended but I never got the reminder email. Can I get a short extension to decide?", "account", "low"),
    ("Please upgrade my seat count from {n} to {n2}. Do I need to do anything on my end?", "account", "low"),
    # ---- delivery ----
    ("My order {order} was supposed to arrive {day} but tracking has not moved in {days} days. Where is it?", "delivery", "high"),
    ("The courier left my parcel {order} at the wrong address. The photo shows a different house entirely.", "delivery", "high"),
    ("Tracking for {order} says delivered but I never received anything. I was home all day.", "delivery", "urgent"),
    ("Can I change the delivery address for order {order}? I entered the wrong postcode by mistake.", "delivery", "medium"),
    ("My parcel arrived damaged — the box was crushed and one item is broken. Order {order}.", "delivery", "high"),
    ("The driver did not ring the bell and now my order {order} is going back to the depot. Can you rebook?", "delivery", "medium"),
    ("I paid for express delivery on {order} but it arrived three days late. I would like the express fee refunded.", "delivery", "medium"),
    ("When will my back-ordered item ship? Order {order} placed {days} days ago, still no dispatch email.", "delivery", "low"),
    ("The tracking page for {order} shows an exception — customs hold. Do I need to do anything?", "delivery", "medium"),
    ("My subscription box has not arrived this month and I was charged. Please check order {order}.", "delivery", "high"),
    # ---- returns ----
    ("I would like to return the {product} from order {order}. It does not fit as described on your site.", "returns", "medium"),
    ("How do I start a return? Order {order}, item arrived faulty — it will not power on at all.", "returns", "high"),
    ("I sent my return back {days} days ago and have heard nothing. Tracking shows it arrived. Where is my refund of {amount}?", "returns", "high"),
    ("Can I exchange the {product} for a larger size instead of a refund? Order {order}.", "returns", "low"),
    ("Your returns label link is broken. Please send a new one for order {order} so I can post it back.", "returns", "medium"),
    ("I was told the refund for {order} would take 5 days. It has been {days} days and nothing has arrived.", "returns", "high"),
    ("The item I received is not what I ordered — wrong colour and model. Order {order}. I need a replacement.", "returns", "high"),
    ("Do you cover return postage for faulty items? My {product} stopped working within a week.", "returns", "medium"),
    ("I missed the 30-day return window by two days because of a hospital stay. Is there anything you can do?", "returns", "low"),
    ("Please cancel the return I requested for order {order} — I have decided to keep the item after all.", "returns", "low"),
    # ---- product_info ----
    ("Does the {product} work with {os}? I need to know before I buy.", "product_info", "low"),
    ("What is the difference between the standard and pro versions of {product}? Which do you recommend for a team of {n}?", "product_info", "low"),
    ("Is there a free trial for the business plan? How long does it last and what are the limits?", "product_info", "low"),
    ("What is the battery life of the {product} under normal use? Your site lists two different figures.", "product_info", "low"),
    ("Can I use {product} offline, or does it need a constant internet connection?", "product_info", "low"),
    ("Do you offer student or education discounts? I am at {uni} and need {n} licences.", "product_info", "low"),
    ("What warranty comes with the {product}? Can I extend it at purchase?", "product_info", "low"),
    ("Is my data stored in the UK or EU? We have compliance requirements to meet.", "product_info", "medium"),
    ("Does {product} integrate with {tool}? We use it across the whole company.", "product_info", "low"),
    ("What are your opening hours for phone support over the holidays?", "product_info", "low"),
    # ---- complaint ----
    ("I have called three times about {order} and nobody calls back. This level of service is unacceptable.", "complaint", "urgent"),
    ("Your support agent was rude to me on the phone today and hung up. I want this escalated to a manager.", "complaint", "urgent"),
    ("I was promised a callback {day} and never got one. Then I was promised a refund that never arrived. Enough.", "complaint", "urgent"),
    ("This is the fourth time I have reported the same billing error. Nobody seems to read the previous tickets.", "complaint", "high"),
    ("Your courier left my parcel in the rain even though I have a covered porch. The box is soaked through.", "complaint", "high"),
    ("I feel misled — the checkout price was {amount} but my card was charged {amount2}. That is not honest.", "complaint", "high"),
    ("Waiting {days} minutes on hold only to be transferred and cut off is not the service I pay for.", "complaint", "high"),
    ("A staff member shared my phone number with a third party without consent. I am considering a formal complaint.", "complaint", "urgent"),
    ("The product description promised features the {product} simply does not have. I feel cheated.", "complaint", "medium"),
    ("Every interaction with your company takes weeks to resolve. I am seriously considering switching providers.", "complaint", "high"),
    # ---- feedback ----
    ("Just wanted to say the new {feature} is fantastic — it saved our team hours this week. Great work!", "feedback", "low"),
    ("The onboarding guide was really clear. I was up and running in under an hour. Thanks!", "feedback", "low"),
    ("Your support agent {name} was brilliant — solved my issue in five minutes. Please pass on my thanks.", "feedback", "low"),
    ("Love the new dashboard design. Much easier to find things now.", "feedback", "low"),
    ("Suggestion: it would be great if the export feature supported CSV as well as PDF.", "feedback", "low"),
    ("The mobile app is so much faster since the last update. Noticeable improvement.", "feedback", "low"),
    ("It would help if the {feature} page had a short video walkthrough for new users.", "feedback", "low"),
    ("Five stars — delivery was a day early and the driver was really friendly.", "feedback", "low"),
    ("The search filters you added are exactly what we needed. Keep it up!", "feedback", "low"),
    ("Small idea: a dark mode for the desktop app would be lovely for late-night work.", "feedback", "low"),
]

_PRODUCTS = ["wireless headphones", "smart speaker", "4K monitor", "mechanical keyboard",
            "fitness tracker", "robot vacuum", "espresso machine", "noise-cancelling earbuds",
            "portable charger", "air purifier"]
_FEATURES = ["billing page", "reports tab", "search bar", "export button", "settings panel",
             "notifications feed", "analytics dashboard", "invite flow", "calendar view", "API docs"]
_OSES = ["iOS", "Android", "Windows 11", "macOS"]
_TOOLS = ["Slack", "Salesforce", "HubSpot", "Jira", "Teams"]
_COMPANIES = ["Northwind", "Acme Retail", "Brightline", "Kestrel & Co", "Loopwork"]
_NAMES = ["Priya", "Tom", "Aisha", "Daniel", "Mei", "Carlos", "Sofia", "James", "Nadia", "Oliver"]
_UNIS = ["Aston University", "Birmingham City University", "the University of Manchester"]
_DAYS = ["Monday", "Tuesday", "yesterday", "last Friday"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        amount=f"£{rng.randint(9, 499)}.{rng.choice(['00', '49', '99'])}",
        amount2=f"£{rng.randint(9, 499)}.{rng.choice(['00', '49', '99'])}",
        card=rng.choice(["Visa ending 4412", "Mastercard ending 8890", "Amex ending 1005"]),
        inv=f"INV-{rng.randint(10000, 99999)}",
        days=rng.randint(2, 21),
        order=f"ORD-{rng.randint(100000, 999999)}",
        day=rng.choice(_DAYS),
        email=f"user{rng.randint(10, 99)}@example.com",
        email2=f"user{rng.randint(10, 99)}@mailbox.co.uk",
        name=rng.choice(_NAMES),
        n=rng.randint(2, 9),
        n2=rng.randint(10, 50),
        product=rng.choice(_PRODUCTS),
        feature=rng.choice(_FEATURES),
        os=rng.choice(_OSES),
        ver=f"{rng.randint(14, 18)}.{rng.randint(0, 5)}",
        code=f"ERR-{rng.randint(100, 599)}",
        tool=rng.choice(_TOOLS),
        company=rng.choice(_COMPANIES),
        uni=rng.choice(_UNIS),
    )


def generate(n: int = 2000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        template, category, priority = rng.choice(_TEMPLATES)
        text = _fill(template, rng)
        rows.append({
            "id": f"TKT-{i + 1:05d}",
            "text": text,
            "category": category,
            "priority": priority,
        })
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic support tickets.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/tickets.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(n=args.n, seed=args.seed)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "category", "priority"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tickets to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

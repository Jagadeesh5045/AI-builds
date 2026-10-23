"""Generate the synthetic dataset for rag-router.

Outputs in data/:
  corpus.jsonl       - ~60 docs: product pages, support KB articles, company narratives
  products.csv       - 24 products
  orders.csv         - 500 synthetic orders (Jan-Jun 2026)
  router_train.jsonl - 180 labelled queries (60 per route) for router training
  router_eval.jsonl  -  90 labelled queries (30 per route) with relevance judgements
                       and, for SQL queries, the expected numeric answer

Everything is deterministic (seeded RNG). No network, no external data.
"""
import csv
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
rng = random.Random(42)

MONTHS = ["january", "february", "march", "april", "may", "june"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
REGIONS = ["north", "south", "east", "west"]

PRODUCTS = [
    ("P001", "Aurora X9 Wireless Headphones", "WH-1002", "Audio", 189.99),
    ("P002", "VoltEdge 20000mAh Power Bank", "PB-2000", "Accessories", 39.99),
    ("P003", "ErgoType Split Keyboard", "KB-2041", "Computing", 129.50),
    ("P004", "LumenDesk LED Monitor Light", "ML-1105", "Accessories", 54.00),
    ("P005", "Stratos Ultralight Laptop 14", "LT-8814", "Computing", 999.00),
    ("P006", "PulseFit Smartwatch S2", "SW-3302", "Wearables", 229.99),
    ("P007", "Nimbus Mesh Office Chair", "CH-4410", "Furniture", 349.00),
    ("P008", "EchoDot Mini Smart Speaker", "SP-5501", "Smart Home", 49.99),
    ("P009", "TerraPro 4K Action Camera", "AC-7712", "Cameras", 279.00),
    ("P010", "GlidePro Wireless Mouse", "MS-2210", "Computing", 59.99),
    ("P011", "HaloRing Video Doorbell", "DB-6610", "Smart Home", 149.99),
    ("P012", "AeroPods Pro Earbuds", "EB-9903", "Audio", 149.00),
    ("P013", "Vertex 27in 4K Monitor", "MN-2714", "Computing", 429.99),
    ("P014", "CloudRest Memory Foam Pillow", "PL-1201", "Home", 34.99),
    ("P015", "IronClad Mechanical Keyboard", "KB-2077", "Computing", 159.00),
    ("P016", "SolarFlare Outdoor Camera", "SC-8810", "Smart Home", 119.99),
    ("P017", "JetStream Handheld Vacuum", "VC-3315", "Home", 139.00),
    ("P018", "OrbitView Webcam 4K", "WC-4402", "Computing", 89.99),
    ("P019", "ThunderBolt 65W GaN Charger", "CH-6501", "Accessories", 45.00),
    ("P020", "Zenith Noise Cancelling Headphones", "WH-1010", "Audio", 249.99),
    ("P021", "PixelPen Stylus Gen 3", "ST-3301", "Accessories", 79.00),
    ("P022", "DuraCell Power Station 500", "PS-5001", "Smart Home", 399.00),
    ("P023", "FlexStand Laptop Riser", "LS-2214", "Accessories", 42.50),
    ("P024", "VistaCam Baby Monitor", "BM-1102", "Smart Home", 99.99),
]

PRODUCT_BLURBS = {
    "P001": "Flagship over-ear headphones with adaptive noise cancelling, 40-hour battery life and multipoint Bluetooth 5.3. Ideal for travel and open offices.",
    "P002": "High-capacity portable charger with 65W USB-C fast charging, airline-safe 20000mAh cell and a digital charge display.",
    "P003": "Ergonomic split keyboard with tenting, low-force tactile switches and a detachable cushioned wrist rest for all-day typing comfort.",
    "P004": "Clip-on monitor light bar with stepless dimming, three colour temperatures and zero screen glare for late-night work.",
    "P005": "Featherweight 990g aluminium laptop with a 14-inch 2.8K OLED display, 18-hour battery and silent fanless cooling.",
    "P006": "Fitness smartwatch with dual-band GPS, heart-rate and sleep tracking, and a 10-day battery in normal use.",
    "P007": "Breathable mesh task chair with dynamic lumbar support, 4D armrests and a weight-sensitive recline mechanism.",
    "P008": "Compact voice-controlled speaker with room-filling sound, smart-home hub built in and multiroom pairing.",
    "P009": "Rugged waterproof action camera recording 4K at 120fps with HyperSmooth stabilisation and a 2-hour swappable battery.",
    "P010": "Silent-click wireless mouse with an ergonomic sculpted shape, 4000 DPI sensor and 70-day battery life.",
    "P011": "Battery video doorbell with 2K HDR, person detection zones, two-way talk and free local storage on the chime hub.",
    "P012": "True wireless earbuds with hybrid noise cancelling, wireless charging case and six microphones for clear calls.",
    "P013": "Colour-accurate 27-inch 4K IPS monitor covering 99% sRGB, height-adjustable stand and USB-C power delivery.",
    "P014": "Cooling gel-infused memory foam pillow with a washable bamboo cover, shaped for side and back sleepers.",
    "P015": "Hot-swappable mechanical keyboard with gasket mount, PBT keycaps and south-facing RGB for deep, soft keystrokes.",
    "P016": "Solar-powered outdoor security camera with colour night vision, AI person detection and no wiring required.",
    "P017": "Cordless handheld vacuum with 25kPa suction, HEPA filtration and a 40-minute runtime for cars and stairs.",
    "P018": "4K webcam with auto-framing, dual noise-cancelling microphones and a physical privacy shutter for video calls.",
    "P019": "Compact gallium-nitride wall charger delivering 65W over USB-C, with foldable prongs for travel.",
    "P020": "Premium noise cancelling headphones with studio-tuned 40mm drivers, 60-hour battery and LDAC hi-res audio.",
    "P021": "Pressure-sensitive stylus with tilt support, magnetic attach and 12-hour battery for drawing and note-taking.",
    "P022": "Portable 500Wh power station with pure sine wave output, solar input and enough capacity for a weekend off-grid.",
    "P023": "Aluminium laptop riser with six height angles, fold-flat design and ventilation cut-outs to keep devices cool.",
    "P024": "HD baby monitor with night vision, temperature sensor, lullabies and a 1000ft long-range parent unit.",
}

KB_DOCS = [
    ("kb_returns", "Returns policy",
     "You can return any unused item within 30 days of delivery for a full refund. Opened electronics can be returned within 14 days provided all accessories and the original packaging are included. Refunds are issued to the original payment method within 5 working days of the warehouse receiving the item. Items marked final sale, gift cards and opened personal-care products cannot be returned."),
    ("kb_warranty", "Warranty terms",
     "Every product ships with a 2-year warranty covering manufacturing defects. Batteries are covered for 12 months. The warranty does not cover accidental damage, water ingress on non-waterproof models, or normal wear such as battery capacity decline. To make a claim, contact support with your order number and a photo of the fault; approved claims are repaired or replaced within 10 working days."),
    ("kb_shipping", "Shipping and delivery",
     "Standard delivery is free on orders over 50 pounds and arrives in 3 to 5 working days. Express delivery costs 6.99 and arrives next working day if you order before 2pm. We deliver to the UK mainland, Northern Ireland and the Channel Islands; remote postcodes may add 2 days. If a parcel arrives damaged, photograph it and report it within 48 hours for a free replacement."),
    ("kb_account", "Account and billing",
     "You can update your address, payment methods and marketing preferences from the account dashboard at any time. VAT invoices are available for every order under order history. If a payment fails, check that your card supports 3D Secure and that the billing postcode matches; you can retry payment from the order page for up to 7 days before the order is cancelled."),
    ("kb_troubleshoot_headphones", "Headphones not charging",
     "If your headphones will not charge, first try a different USB-C cable and a wall charger rated 5V. Clean the charging port gently with a dry cotton bud, as pocket lint is the most common cause. Hold the power button for 15 seconds to reset the device, then charge for at least 30 minutes before testing. If the LED still does not light, contact support for a warranty replacement."),
    ("kb_troubleshoot_keyboard", "Keyboard keeps disconnecting",
     "A keyboard that drops connection is usually caused by wireless interference or low battery. Move the USB receiver to a front port away from metal objects, replace or recharge the batteries, and keep the keyboard within 10 metres of the receiver. On Bluetooth models, remove the device from your paired list and re-pair it. Updating the firmware from the support page fixes most remaining cases."),
    ("kb_student_discount", "Student discount",
     "Students with a verified university email receive 10 percent off full-price items. Verify once per academic year through the student portal; the discount applies automatically at checkout. It cannot be combined with other promo codes or sale prices, and excludes gift cards and final-sale items."),
    ("kb_tradein", "Trade-in programme",
     "Trade in your old headphones, laptops or smartwatches for store credit. Answer a short condition questionnaire online, post the device free of charge, and receive credit within 7 days of inspection. Devices must power on and have no cracked screens for the standard valuation; accessories increase the offer."),
    ("kb_giftcards", "Gift cards",
     "Gift cards are delivered by email within minutes and never expire. They can be used across multiple orders and combined with promo codes. Lost gift card emails can be re-sent from your account; treat the code like cash, as we cannot replace cards redeemed by someone else."),
    ("kb_install_doorbell", "Installing the video doorbell",
     "Charge the doorbell fully before mounting. Use the included wedge to angle the camera toward your path, mount at about 1.2 metres high, and connect to 2.4GHz Wi-Fi during setup. The chime hub should be plugged in within 15 metres. Run the in-app signal test; two bars or more gives reliable motion alerts."),
    ("kb_data_privacy", "Privacy and your data",
     "We collect only the data needed to fulfil orders and improve the store: contact details, order history and anonymised browsing analytics. We never sell personal data. You can download or delete your data at any time from the privacy dashboard, and marketing emails stop immediately when you unsubscribe."),
    ("kb_speaker_setup", "Setting up the smart speaker",
     "Plug in the speaker and wait for the pulsing light, then add it in the companion app over 2.4GHz Wi-Fi. Link your music accounts to enable voice playback, and create speaker groups for multiroom audio. If setup stalls, restart your router and keep the phone within 3 metres during pairing."),
]

NARRATIVE_DOCS = [
    ("co_sustainability", "Our sustainability programme",
     "Since 2023 we have cut packaging plastic by 82 percent by switching to moulded fibre and paper tape. Delivery routes are optimised nightly to reduce mileage, and 61 percent of last-mile deliveries in London are now made by electric van or cargo bike. Returned electronics are refurbished and resold rather than landfilled, and our repair centres fixed over 40,000 devices last year."),
    ("co_recommendations", "How our recommendation engine works",
     "Product recommendations blend collaborative filtering over anonymised purchase histories with content signals such as category, price band and brand affinity. A nightly batch job retrains the model, while a real-time layer re-ranks suggestions based on the current browsing session. Customers can opt out of personalisation entirely from the privacy dashboard."),
    ("co_supplychain", "Supply chain and sourcing",
     "We source from 140 audited suppliers across 18 countries. Every supplier signs our code of conduct covering fair wages, working hours and conflict minerals, and high-risk factories are audited annually by an independent third party. Key components are dual-sourced so that a disruption at one site never halts production."),
    ("co_story_founder", "Why we started the company",
     "The company began in 2019 when the two founders could not find affordable, repairable tech accessories. They started by refurbishing returned electronics in a Birmingham garage, selling them with a proper warranty. That repair-first philosophy still shapes the business: every product page shows a repairability score and spare parts are stocked for 7 years."),
    ("co_charity", "Community and charity work",
     "Each quarter, staff vote for a local charity to receive 1 percent of profits. Recent partners include food banks in Birmingham and Manchester and a digital-inclusion project donating refurbished laptops to schools. Employees get two paid volunteering days per year."),
    ("co_ai_support", "AI in customer support",
     "Routine support questions are first handled by an assistant that retrieves answers from the help centre and drafts replies for human agents to approve. The assistant resolves about 35 percent of chats end to end, and every automated answer carries a feedback button whose ratings retrain the system weekly. Complex or sensitive cases always go straight to a human."),
]

VECTOR_QUERY_POOL = [  # (question, relevant doc ids)
    ("Can I return headphones after I have opened the box?", ["kb_returns"]),
    ("What is the returns window for opened electronics?", ["kb_returns"]),
    ("How long do I have to send something back for a refund?", ["kb_returns"]),
    ("My parcel arrived smashed. What should I do?", ["kb_shipping"]),
    ("How much does express delivery cost and when does it arrive?", ["kb_shipping"]),
    ("Do you deliver to Northern Ireland?", ["kb_shipping"]),
    ("What does the warranty cover for batteries?", ["kb_warranty"]),
    ("How do I make a warranty claim for a faulty device?", ["kb_warranty"]),
    ("Is accidental damage covered by the warranty?", ["kb_warranty"]),
    ("My headphones will not charge. Any fixes?", ["kb_troubleshoot_headphones"]),
    ("The charging light on my earbuds never comes on. What now?", ["kb_troubleshoot_headphones"]),
    ("My wireless keyboard keeps dropping its connection. Help?", ["kb_troubleshoot_keyboard"]),
    ("How do I stop my Bluetooth keyboard disconnecting?", ["kb_troubleshoot_keyboard"]),
    ("Do students get a discount, and how do I claim it?", ["kb_student_discount"]),
    ("Can I combine the student discount with a promo code?", ["kb_student_discount"]),
    ("How does the trade-in programme value my old laptop?", ["kb_tradein"]),
    ("What condition does my old smartwatch need to be in to trade it in?", ["kb_tradein"]),
    ("Do gift cards expire?", ["kb_giftcards"]),
    ("Can I use a gift card together with a discount code?", ["kb_giftcards"]),
    ("How high should I mount the video doorbell?", ["kb_install_doorbell"]),
    ("Which Wi-Fi band does the doorbell need during setup?", ["kb_install_doorbell", "kb_speaker_setup"]),
    ("Do you sell my personal data to advertisers?", ["kb_data_privacy"]),
    ("How do I delete my account data?", ["kb_data_privacy"]),
    ("How do I set up multiroom audio on the smart speaker?", ["kb_speaker_setup"]),
    ("What should I do if speaker setup stalls halfway?", ["kb_speaker_setup"]),
    ("How has the company reduced plastic packaging?", ["co_sustainability"]),
    ("What share of London deliveries are electric?", ["co_sustainability"]),
    ("How do product recommendations on the site work?", ["co_recommendations"]),
    ("Can I turn off personalised recommendations?", ["co_recommendations", "kb_data_privacy"]),
    ("How do you audit your suppliers?", ["co_supplychain"]),
    ("What happens to returned electronics?", ["co_sustainability", "co_story_founder"]),
    ("Which charities did you support recently?", ["co_charity"]),
    ("How many devices did your repair centres fix last year?", ["co_sustainability"]),
    ("What percentage of support chats does the AI assistant resolve?", ["co_ai_support"]),
    ("Where can I find a VAT invoice for my order?", ["kb_account"]),
    ("My card payment failed. What should I check?", ["kb_account"]),
]

KEYWORD_QUERY_POOL = [  # (question, relevant product ids)
    ("Aurora X9 Wireless Headphones battery life", ["P001"]),
    ("What is the price of the VoltEdge 20000mAh Power Bank?", ["P002"]),
    ("ErgoType Split Keyboard wrist rest included?", ["P003"]),
    ("SKU KB-2041 specifications", ["P003"]),
    ("LumenDesk LED Monitor Light colour temperatures", ["P004"]),
    ("Stratos Ultralight Laptop 14 weight", ["P005"]),
    ("PulseFit Smartwatch S2 GPS accuracy", ["P006"]),
    ("Nimbus Mesh Office Chair lumbar support", ["P007"]),
    ("EchoDot Mini Smart Speaker multiroom pairing", ["P008"]),
    ("TerraPro 4K Action Camera waterproof depth", ["P009"]),
    ("GlidePro Wireless Mouse DPI settings", ["P010"]),
    ("HaloRing Video Doorbell local storage", ["P011"]),
    ("AeroPods Pro Earbuds noise cancelling", ["P012"]),
    ("Vertex 27in 4K Monitor sRGB coverage", ["P013"]),
    ("CloudRest Memory Foam Pillow washable cover", ["P014"]),
    ("IronClad Mechanical Keyboard hot swappable switches", ["P015"]),
    ("SolarFlare Outdoor Camera solar panel charging", ["P016"]),
    ("JetStream Handheld Vacuum runtime minutes", ["P017"]),
    ("OrbitView Webcam 4K privacy shutter", ["P018"]),
    ("ThunderBolt 65W GaN Charger foldable prongs", ["P019"]),
    ("Zenith Noise Cancelling Headphones LDAC support", ["P020"]),
    ("PixelPen Stylus Gen 3 tilt support", ["P021"]),
    ("DuraCell Power Station 500 solar input watts", ["P022"]),
    ("FlexStand Laptop Riser height angles", ["P023"]),
    ("VistaCam Baby Monitor range in feet", ["P024"]),
    ("WH-1002 vs WH-1010 differences", ["P001", "P020"]),
    ("cheapest 65W USB-C charger", ["P019", "P002"]),
    ("ergonomic keyboard with tenting", ["P003"]),
    ("4K webcam for video calls", ["P018"]),
    ("power bank with digital display", ["P002"]),
]

SQL_TEMPLATES = [
    # (template, kind) - kind in {sum_revenue_month, count_orders_month, sum_units_product,
    #                            avg_order_region, top_product_month, sum_revenue_region_month,
    #                            count_orders_product_month}
    ("What was the total revenue in {month}?", "sum_revenue_month"),
    ("How much sales revenue did we make in {month}?", "sum_revenue_month"),
    ("Tell me the total revenue for {month}.", "sum_revenue_month"),
    ("How many orders were placed in {month}?", "count_orders_month"),
    ("What is the order count for {month}?", "count_orders_month"),
    ("Number of orders in {month}?", "count_orders_month"),
    ("How many units of {product} were sold?", "sum_units_product"),
    ("What is the total quantity sold of {product}?", "sum_units_product"),
    ("Units sold for {product}?", "sum_units_product"),
    ("What is the average order value in the {region} region?", "avg_order_region"),
    ("Average order value for {region}?", "avg_order_region"),
    ("Which product had the highest revenue in {month}?", "top_product_month"),
    ("What was the best-selling product by revenue in {month}?", "top_product_month"),
    ("Top product by revenue in {month}?", "top_product_month"),
    ("What was the total revenue from {region} in {month}?", "sum_revenue_region_month"),
    ("Revenue from the {region} region in {month}?", "sum_revenue_region_month"),
    ("How many orders did {product} get in {month}?", "count_orders_product_month"),
    ("Order count for {product} in {month}?", "count_orders_product_month"),
]

# Extra rewordings used ONLY for training, so the router sees varied phrasing.
SQL_TRAIN_PARAPHRASES = [
    ("In {month}, what did total revenue come to?", "sum_revenue_month"),
    ("Sum up the revenue for {month}.", "sum_revenue_month"),
    ("Give me the revenue total for {month}.", "sum_revenue_month"),
    ("Count the orders from {month}.", "count_orders_month"),
    ("How many orders did we get in {month}?", "count_orders_month"),
    ("Total orders placed during {month}?", "count_orders_month"),
    ("How many {product} units have we sold in total?", "sum_units_product"),
    ("What are the lifetime unit sales of {product}?", "sum_units_product"),
    ("For the {region} region, what is the mean order value?", "avg_order_region"),
    ("Mean order value in {region}?", "avg_order_region"),
    ("What is the typical order value for {region} customers?", "avg_order_region"),
    ("In {month}, which product earned the most revenue?", "top_product_month"),
    ("Highest-grossing product in {month}?", "top_product_month"),
    ("How much revenue came from {region} during {month}?", "sum_revenue_region_month"),
    ("What did the {region} region bring in during {month}?", "sum_revenue_region_month"),
    ("In {month}, how many orders included {product}?", "count_orders_product_month"),
]


def write_products():
    with open(DATA / "products.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "name", "sku", "category", "price"])
        w.writerows(PRODUCTS)


def write_orders():
    with open(DATA / "orders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "product_id", "customer_id", "quantity", "unit_price", "order_date", "region"])
        for i in range(1, 501):
            pid, _, _, _, price = rng.choice(PRODUCTS)
            qty = rng.randint(1, 5)
            month = rng.randint(1, 6)
            day = rng.randint(1, 28)
            w.writerow([
                f"O{i:04d}", pid, f"C{rng.randint(1, 120):03d}", qty,
                price, f"2026-{month:02d}-{day:02d}", rng.choice(REGIONS).capitalize(),
            ])


def write_corpus():
    docs = []
    for pid, name, sku, category, price in PRODUCTS:
        docs.append({
            "doc_id": pid,
            "title": name,
            "text": f"{name} (SKU {sku}) - {category}. Price: £{price:.2f}. {PRODUCT_BLURBS[pid]}",
        })
    for doc_id, title, text in KB_DOCS + NARRATIVE_DOCS:
        docs.append({"doc_id": doc_id, "title": title, "text": text})
    with open(DATA / "corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"corpus: {len(docs)} docs")


def split_pool(pool, n_eval):
    """Deterministic split of a query pool into train/eval portions."""
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    eval_idx = set(idx[:n_eval])
    train = [q for i, q in enumerate(pool) if i not in eval_idx]
    ev = [q for i, q in enumerate(pool) if i in eval_idx]
    return train, ev


def sql_queries(n, seed, templates):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        template, kind = r.choice(templates)
        month = r.choice(MONTHS)
        product = r.choice(PRODUCTS)
        region = r.choice(REGIONS)
        q = template.format(month=month.capitalize(), product=product[1], region=region.capitalize())
        out.append((q, kind, {"month": month, "product_id": product[0],
                             "product_name": product[1], "region": region}))
    return out


def expand_train(pairs, seed):
    """Add realistic wrapper variants to training queries (eval untouched)."""
    r = random.Random(seed)
    wrappers = [
        lambda q: q,
        lambda q: "Can you tell me: " + q[0].lower() + q[1:],
        lambda q: "Quick question - " + q[0].lower() + q[1:],
        lambda q: q.rstrip("?") + ", please?",
    ]
    out = []
    for q, rel in pairs:
        out.append((q, rel))
        w = r.choice(wrappers[1:])
        out.append((w(q), rel))
    return out


def write_router_sets():
    v_train, v_eval = split_pool(VECTOR_QUERY_POOL, 12)
    k_train, k_eval = split_pool(KEYWORD_QUERY_POOL, 12)
    v_train = expand_train(v_train, seed=11)
    k_train = expand_train(k_train, seed=12)
    s_train = sql_queries(60, seed=7, templates=SQL_TEMPLATES + SQL_TRAIN_PARAPHRASES)
    s_eval = sql_queries(30, seed=99, templates=SQL_TEMPLATES)

    with open(DATA / "router_train.jsonl", "w") as f:
        for q, _ in v_train:
            f.write(json.dumps({"query": q, "label": "vector"}) + "\n")
        for q, _ in k_train:
            f.write(json.dumps({"query": q, "label": "keyword"}) + "\n")
        for q, kind, params in s_train:
            f.write(json.dumps({"query": q, "label": "sql"}) + "\n")

    with open(DATA / "router_eval.jsonl", "w") as f:
        for q, rel in v_eval:
            f.write(json.dumps({"query": q, "label": "vector",
                                "relevant_doc_ids": rel, "k": 5}) + "\n")
        for q, rel in k_eval:
            f.write(json.dumps({"query": q, "label": "keyword",
                                "relevant_doc_ids": rel, "k": 5}) + "\n")
        for q, kind, params in s_eval:
            f.write(json.dumps({"query": q, "label": "sql", "kind": kind,
                                "params": params}) + "\n")
    print(f"train queries: {len(v_train) + len(k_train) + len(s_train)}, "
          f"eval queries: {len(v_eval) + len(k_eval) + len(s_eval)}")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    write_products()
    write_orders()
    write_corpus()
    write_router_sets()
    print("data written to", DATA)

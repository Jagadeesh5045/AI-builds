"""QA-bench-style bug hunting on a synthetic running web app.

Momentic's QA Bench (published 3 Nov 2026, covered the same day) measures how
many real bugs an agent finds when pointed at a *running* web application:
1,396 human-labelled bugs across four open-source apps (Gitea, Metabase,
Twenty, Saleor), 20 product areas, each configuration on a fixed 60-minute
budget rather than a fixed token budget. Its headline figures are the
vendor's own (Momentic built the benchmark and its agent Mo tops it, and the
coverage notes they were not independently reproduced), so this demo does
not re-derive them: it re-implements the *measurement idea* on a small
synthetic shop app with 12 planted, labelled bugs, and asks the question
QA Bench asks: under a fixed time budget, which hunting strategy finds the
most real bugs, at what precision, and at what cost per bug?

Strategies
----------
* SystematicHunter: runs a written spec checklist, probe by probe.
* PrioritisedHunter: runs the same checklist ordered by the bug density
  seen on a labelled dev split (area-level history, not the test bugs).
* RandomHunter: wanders the app with seeded random actions and files a
  report whenever an action disagrees with the spec oracle it carries.
  It can also file false reports (it mis-reads a spec edge case) and
  duplicate reports (the same bug found twice).

Scoring mirrors QA Bench's table: bugs matched, recall against the
labelled set, precision over reports filed, and cost per matched bug.

Everything here is stdlib-only and deterministic (RandomHunter is seeded).
The app's bugs are deliberate and labelled in BUGS; a probe is a spec
check, not a search for a known bug id, so a hunter that covers an area
finds the bugs in it whether or not it "knows" they are there.
"""

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- app

PRODUCTS = [
    {"id": 1, "name": "Widget", "price": 10.00, "stock": 5},
    {"id": 2, "name": "widget pro", "price": 20.00, "stock": 0},
    {"id": 3, "name": "Gadget", "price": 7.50, "stock": 3},
    {"id": 4, "name": "Doohickey", "price": 3.25, "stock": 9},
    {"id": 5, "name": "Thingamajig", "price": 99.99, "stock": 1},
    {"id": 6, "name": "Contraption", "price": 42.00, "stock": 4},
]


@dataclass
class Bug:
    bug_id: str
    area: str
    description: str


BUGS = [
    Bug("B01", "catalogue", "page 2 repeats the last item of page 1"),
    Bug("B02", "search", "search is case-sensitive, so 'Widget' misses 'widget pro'"),
    Bug("B03", "search", "empty query returns nothing instead of all products"),
    Bug("B04", "catalogue", "price sort ascending actually returns descending order"),
    Bug("B05", "catalogue", "in-stock filter includes an out-of-stock product"),
    Bug("B06", "cart", "adding the same item twice creates a duplicate line"),
    Bug("B07", "cart", "setting quantity to 0 leaves the line in the cart"),
    Bug("B08", "cart", "tax is floored to whole pounds instead of rounded to pence"),
    Bug("B09", "checkout", "free shipping applies at >100, not >=100 (boundary missed)"),
    Bug("B10", "checkout", "total is summed in pounds, then labelled as pence"),
    Bug("B11", "account", "email validation accepts an address with no TLD"),
    Bug("B12", "search", "search matches internal supplier codes it should never look at"),
]


class ShopApp:
    """A small shop with exactly the 12 bugs in BUGS, and nothing else."""

    def __init__(self):
        self.cart = []  # list of (product_id, qty)

    # -- catalogue ------------------------------------------------------
    def list_page(self, page, size=2):
        # B01: slice starts one item too early on every page after the first.
        start = 0 if page == 1 else (page - 1) * size - 1
        return PRODUCTS[start:start + size]

    def sorted_by_price(self):
        # B04: descending, although the spec says ascending.
        return sorted(PRODUCTS, key=lambda p: p["price"], reverse=True)

    def in_stock(self):
        # B05: stock >= 0 admits the stock-0 product.
        return [p for p in PRODUCTS if p["stock"] >= 0]

    # -- search ---------------------------------------------------------
    def search(self, query):
        # B03: empty query returns nothing.
        if query == "":
            return []
        out = []
        for p in PRODUCTS:
            # B02: case-sensitive name match.
            # B12: also matches the internal supplier code, which the
            # spec says search must never look at.
            if query in p["name"] or query == f"SUP-{p['id']}":
                out.append(p)
        return out

    # -- cart -----------------------------------------------------------
    def add_to_cart(self, product_id, qty=1):
        # B06: never merges with an existing line.
        self.cart.append((product_id, qty))

    def set_quantity(self, product_id, qty):
        # B07: a 0 quantity line is kept instead of removed.
        self.cart = [(pid, q if pid != product_id else qty)
                     for pid, q in self.cart]

    def cart_total(self):
        subtotal = sum(
            next(p["price"] for p in PRODUCTS if p["id"] == pid) * q
            for pid, q in self.cart)
        # B08: tax floored to whole pounds.
        tax = math.floor(subtotal * 0.20)
        return subtotal + tax

    # -- checkout -------------------------------------------------------
    def shipping_cost(self, subtotal):
        # B09: strictly greater than 100, so exactly 100 still pays.
        return 0.0 if subtotal > 100 else 5.0

    def checkout_total_pence(self):
        # B10: pounds summed, then labelled pence without converting.
        return self.cart_total()

    # -- account --------------------------------------------------------
    def valid_email(self, email):
        # B11: only requires an '@' and a dot somewhere before it.
        return "@" in email and "." in email.split("@")[0]


# ---------------------------------------------------------------- probes
# A probe is a spec check: it returns True when the app behaves as the
# spec says. A failing probe files one report for the bug it covers.

@dataclass
class Probe:
    name: str
    area: str
    bug_id: str  # "" for a healthy-behaviour probe with no planted bug
    check: object = None


def make_probes():
    def page2():
        ids = [p["id"] for p in ShopApp().list_page(2, size=2)]
        return ids == [3, 4]

    def search_case():
        names = [p["name"] for p in ShopApp().search("Widget")]
        return "widget pro" in names

    def search_empty():
        return len(ShopApp().search("")) == len(PRODUCTS)

    def sort_asc():
        prices = [p["price"] for p in ShopApp().sorted_by_price()]
        return prices == sorted(prices)

    def stock_filter():
        return all(p["stock"] > 0 for p in ShopApp().in_stock())

    def merge_lines():
        app = ShopApp()
        app.add_to_cart(1)
        app.add_to_cart(1)
        return len(app.cart) == 1

    def zero_removes():
        app = ShopApp()
        app.add_to_cart(1)
        app.set_quantity(1, 0)
        return len(app.cart) == 0

    def tax_rounding():
        app = ShopApp()
        app.add_to_cart(3, 3)  # 22.50 subtotal, 20% tax = 4.50 exactly
        app.add_to_cart(4)     # +3.25 = 25.75, tax 5.15
        return abs(app.cart_total() - (25.75 + 5.15)) < 1e-9

    def free_shipping_boundary():
        return ShopApp().shipping_cost(100.0) == 0.0

    def pence_total():
        app = ShopApp()
        app.add_to_cart(1)  # 10.00 pounds -> 1000 pence + tax
        return app.checkout_total_pence() >= 1000

    def email_tld():
        # Dot before the '@' should not count: a TLD dot is required after.
        return not ShopApp().valid_email("first.last@com")

    def search_scope():
        # Supplier codes are internal: searching one must return nothing.
        return ShopApp().search("SUP-3") == []

    def healthy_price():
        return ShopApp().list_page(1, size=2)[0]["price"] == 10.00

    def healthy_page_size():
        return len(ShopApp().list_page(1, size=3)) == 3

    def healthy_search_exact():
        names = [p["name"] for p in ShopApp().search("Gadget")]
        return names == ["Gadget"]

    def healthy_cart_line():
        app = ShopApp()
        app.add_to_cart(2)
        return len(app.cart) == 1 and app.cart[0][0] == 2

    def healthy_shipping_paid():
        return ShopApp().shipping_cost(50.0) == 5.0

    def healthy_email_accepted():
        return ShopApp().valid_email("first.last@example.com")

    # Spec-document order: healthy and buggy checks interleaved, the way
    # a real checklist is written feature by feature.
    return [
        Probe("first-page-price", "catalogue", "", healthy_price),
        Probe("page-2-items", "catalogue", "B01", page2),
        Probe("sort-price-ascending", "catalogue", "B04", sort_asc),
        Probe("page-size-honoured", "catalogue", "", healthy_page_size),
        Probe("in-stock-filter", "catalogue", "B05", stock_filter),
        Probe("search-exact-name", "search", "", healthy_search_exact),
        Probe("search-case-insensitive", "search", "B02", search_case),
        Probe("search-empty-returns-all", "search", "B03", search_empty),
        Probe("search-name-only", "search", "B12", search_scope),
        Probe("cart-single-line", "cart", "", healthy_cart_line),
        Probe("cart-merges-duplicates", "cart", "B06", merge_lines),
        Probe("quantity-zero-removes", "cart", "B07", zero_removes),
        Probe("tax-rounded-to-pence", "cart", "B08", tax_rounding),
        Probe("shipping-charged-under-100", "checkout", "", healthy_shipping_paid),
        Probe("free-shipping-at-100", "checkout", "B09", free_shipping_boundary),
        Probe("checkout-total-in-pence", "checkout", "B10", pence_total),
        Probe("email-valid-accepted", "account", "", healthy_email_accepted),
        Probe("email-requires-tld", "account", "B11", email_tld),
    ]


# ---------------------------------------------------------------- hunters

@dataclass
class Report:
    bug_id: str  # "" for a false report
    probe: str


@dataclass
class RunResult:
    hunter: str
    minutes: float
    cost: float
    reports: list = field(default_factory=list)

    @property
    def matched(self):
        return sorted({r.bug_id for r in self.reports if r.bug_id})

    @property
    def recall(self):
        return len(self.matched) / len(BUGS)

    @property
    def precision(self):
        if not self.reports:
            return 0.0
        true_reports = len({r.bug_id for r in self.reports if r.bug_id})
        return true_reports / len(self.reports)

    @property
    def cost_per_bug(self):
        return self.cost / len(self.matched) if self.matched else float("inf")

    def summary(self):
        return (f"{self.hunter}: {len(self.matched)}/{len(BUGS)} bugs matched "
                f"(recall {self.recall:.1%}, precision {self.precision:.1%}, "
                f"${self.cost:.2f} total, ${self.cost_per_bug:.2f}/bug, "
                f"{self.minutes:.0f} min)")


PROBE_MINUTES = 2.0
PROBE_COST = 0.40


def run_checklist(probes, budget_minutes, name, cost_per_probe=PROBE_COST):
    """Run probes in the given order until the time budget is spent."""
    reports, minutes, cost = [], 0.0, 0.0
    for probe in probes:
        if minutes + PROBE_MINUTES > budget_minutes:
            break
        minutes += PROBE_MINUTES
        cost += cost_per_probe
        if not probe.check():
            reports.append(Report(probe.bug_id, probe.name))
    return RunResult(name, minutes, cost, reports)


def run_random(seed, budget_minutes, name="RandomHunter"):
    """Seeded random walk: cheap actions, noisy reports, duplicates kept."""
    rng = random.Random(seed)
    probes = make_probes()
    reports, minutes, cost = [], 0.0, 0.0
    while minutes + 0.5 <= budget_minutes:
        minutes += 0.5
        cost += 0.08
        probe = rng.choice(probes)
        if not probe.check():
            reports.append(Report(probe.bug_id, probe.name))
        elif rng.random() < 0.05:
            # Mis-reads a spec edge case: a false report on a healthy probe.
            reports.append(Report("", probe.name + "-misread"))
    return RunResult(name, minutes, cost, reports)


# Bug density per area on a labelled dev split (history, not test bugs):
# search is buggiest here, then catalogue, cart, checkout, account.
DEV_DENSITY = {"search": 4, "catalogue": 3, "cart": 3, "checkout": 2, "account": 1}


def prioritised_order(probes):
    return sorted(probes, key=lambda p: -DEV_DENSITY.get(p.area, 0))


# ---------------------------------------------------------------- headline check

def qa_bench_headline_recall(matched, total=1396):
    """Verify the published QA Bench table is internally consistent."""
    return matched / total


# ---------------------------------------------------------------- demo

def main():
    probes = make_probes()
    print("QA Bench idea, on a synthetic shop (12 labelled bugs, 60-min budget)")
    print("Systematic:", run_checklist(probes, 60, "SystematicHunter").summary())
    print("Prioritised:", run_checklist(
        prioritised_order(probes), 60, "PrioritisedHunter").summary())
    print("Random:     ", run_random(7, 60).summary())
    print()
    print("Same hunters on a tight 12-minute budget (full checklist needs 36 min):")
    print("Systematic:", run_checklist(probes, 12, "SystematicHunter").summary())
    print("Prioritised:", run_checklist(
        prioritised_order(probes), 12, "PrioritisedHunter").summary())
    print("Random:     ", run_random(7, 12).summary())
    print()
    print("Published QA Bench headline arithmetic (vendor figures, 31 Oct exports):")
    for name, matched, reported in [
            ("Mo (Momentic)", 250.3, 0.179),
            ("Codex / GPT-6.1 Sol + Playwright MCP", 94.3, 0.068)]:
        computed = qa_bench_headline_recall(matched)
        print(f"  {name}: {matched}/1396 = {computed:.1%} "
              f"(published {reported:.1%}, consistent: "
              f"{abs(computed - reported) < 0.001})")


if __name__ == "__main__":
    main()

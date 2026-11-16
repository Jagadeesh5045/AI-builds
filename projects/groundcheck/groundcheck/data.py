"""Seeded support knowledge base and evaluation cases.

Six short product support passages. Evaluation answers are labelled:
fully grounded answers should pass, answers with planted unsupported
claims (wrong prices, invented features, wrong limits) should fail.
"""
PASSAGES = [
    "The Pro plan costs 29 pounds per month and includes 10,000 API calls. Overage is billed at 2 pounds per 1,000 extra calls.",
    "Password reset links expire after 30 minutes. Users can request a new link from the sign in page at any time.",
    "Data is retained for 90 days after account closure, then permanently deleted. Exports are available in CSV and JSON format.",
    "The API rate limit is 600 requests per minute per key. Exceeding the limit returns HTTP 429 with a Retry-After header.",
    "Single sign on supports SAML 2.0 and OpenID Connect. SCIM provisioning is available on the Enterprise plan only.",
    "Refunds are issued within 5 working days to the original payment method. Annual plans can be cancelled within 14 days for a full refund.",
]
CASES = [
    ("grounded", "The Pro plan costs 29 pounds per month and includes 10,000 API calls [1].", [0]),
    ("grounded", "Password reset links expire after 30 minutes [2]. Users can request a new link from the sign in page [2].", [1]),
    ("grounded", "The API rate limit is 600 requests per minute per key [4]. Exceeding the limit returns HTTP 429 [4].", [3]),
    ("grounded", "Data is retained for 90 days after account closure, then permanently deleted [3]. Exports are available in CSV and JSON format [3].", [2]),
    ("hallucinated", "The Pro plan costs 9 pounds per month and includes unlimited API calls [1].", [0]),
    ("hallucinated", "Password reset links never expire and support can reset them by phone [2].", [1]),
    ("hallucinated", "Refunds are issued instantly in cash and annual plans can be cancelled within 90 days [6].", [5]),
    ("hallucinated", "Single sign on supports SAML 2.0 and the platform includes a built in video editor [5].", [4]),
]

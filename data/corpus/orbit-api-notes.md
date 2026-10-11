# Orbit API Engineering Notes

Working notes for the Orbit public API. These are the rules the client SDKs assume.

## 1. Transport

### 1.1 Versioning

The version is part of the path, not a header. Additive fields are considered compatible.
Removing a field requires a new major version and a twelve month deprecation window.

### 1.2 Timeouts

Server side work is capped at thirty seconds. Requests that exceed it are terminated and the
client receives no partial body. Long running imports must use the job endpoints instead.

## 2. Request Handling

### 2.1 Idempotency

Writes accept a client supplied key. Replaying the same key within twenty four hours returns the
original response body byte for byte, including the original status code. Keys are scoped to the
credential that created them, so two credentials never collide.

### 2.2 Rate Limiting

Limits are enforced per credential with a token bucket. The steady state allowance is six hundred
requests per minute with a burst of two hundred. Exhaustion returns status 429 and a header naming
the number of seconds to wait. The bucket refills continuously rather than on a fixed boundary.

### 2.3 Pagination

Collections use opaque cursors. The cursor encodes the sort position, so a record inserted during
a traversal is not guaranteed to appear. Page size defaults to fifty and may not exceed five
hundred. Cursors expire after one hour.

## 3. Delivery

### 3.1 Webhook Retries

Delivery attempts follow an exponential schedule with jitter. The first retry waits roughly ten
seconds and the schedule runs for seventy two hours before the message is parked. A parked message
can be replayed from the dashboard for up to thirty days.

### 3.2 Signature Verification

Every payload carries a signature over the raw body and a timestamp. Receivers should reject
anything older than five minutes to blunt replay attacks, and must compare in constant time.

## 4. Failure Modes

### 4.1 Error Taxonomy

Client errors never trigger a retry. The one exception is a conflict on a write that carried an
idempotency key, which is safe to replay unchanged. Server errors are always safe to retry for
read paths and for writes that carried a key.

### 4.2 Partial Failures

Batch endpoints return one result per input position, in the same order as the request. A batch is
not atomic. Callers must inspect each position rather than the envelope status alone.

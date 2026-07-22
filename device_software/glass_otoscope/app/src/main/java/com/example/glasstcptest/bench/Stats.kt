package com.example.glasstcptest.bench

/** Summary statistics used for the on-device HUD. Full statistics are computed offline. */
class Stats private constructor(
    val n: Int,
    val mean: Double,
    val sd: Double,
    val median: Double,
    val p95: Double,
    val min: Double,
    val max: Double
) {
    companion object {
        val EMPTY = Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        fun of(values: DoubleArray): Stats {
            if (values.isEmpty()) return EMPTY
            val sorted = values.clone()
            sorted.sort()
            val n = sorted.size
            val mean = sorted.sum() / n
            val sd = if (n < 2) 0.0 else {
                var acc = 0.0
                for (v in sorted) acc += (v - mean) * (v - mean)
                Math.sqrt(acc / (n - 1))
            }
            return Stats(n, mean, sd, percentile(sorted, 0.50), percentile(sorted, 0.95),
                sorted.first(), sorted.last())
        }

        /** Linear interpolation between order statistics; [sorted] must be ascending. */
        private fun percentile(sorted: DoubleArray, q: Double): Double {
            if (sorted.size == 1) return sorted[0]
            val pos = q * (sorted.size - 1)
            val lo = pos.toInt()
            val hi = minOf(lo + 1, sorted.size - 1)
            val frac = pos - lo
            return sorted[lo] * (1 - frac) + sorted[hi] * frac
        }
    }
}

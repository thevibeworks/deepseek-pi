package stats

// Mean returns the arithmetic mean of xs. An empty slice returns 0.
func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	total := 0.0
	for i := 1; i < len(xs); i++ {
		total += xs[i]
	}
	return total / float64(len(xs))
}

package stats

import "testing"

func TestMean(t *testing.T) {
	tests := []struct {
		name string
		in   []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 5},
		{"several", []float64{1, 2, 3, 4}, 2.5},
		{"negatives", []float64{-2, 2}, 0},
	}
	for _, tt := range tests {
		if got := Mean(tt.in); got != tt.want {
			t.Errorf("%s: Mean(%v) = %v, want %v", tt.name, tt.in, got, tt.want)
		}
	}
}

BINARY := deepseek-pi
PKG    := ./cmd/deepseek-pi
BIN    := bin/$(BINARY)

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.buildVersion=$(VERSION)

.DEFAULT_GOAL := build

.PHONY: build
build: ## Build the binary into ./bin
	go build -ldflags '$(LDFLAGS)' -o $(BIN) $(PKG)

.PHONY: install
install: ## Install into GOBIN
	go install -ldflags '$(LDFLAGS)' $(PKG)

.PHONY: test
test: ## Unit tests (no network)
	go test ./...

.PHONY: test-race
test-race: ## Unit tests under the race detector
	go test -race ./...

.PHONY: test-live
test-live: ## Tests that hit the real API; needs DEEPSEEK_API_KEY. Costs a few cents.
	DEEPSEEK_PI_LIVE=1 go test ./... -run TestLive -v -count=1

.PHONY: cover
cover: ## Coverage report
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

.PHONY: lint
lint: ## golangci-lint
	golangci-lint run

.PHONY: fmt
fmt: ## Format and tidy
	gofmt -w .
	go mod tidy

.PHONY: check
check: fmt vet test ## Everything CI runs

.PHONY: vet
vet: ## go vet
	go vet ./...

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf bin dist coverage.out

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

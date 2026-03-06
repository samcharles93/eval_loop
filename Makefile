VERSION := $(shell git describe --tags --always 2>/dev/null || echo "dev")
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS := -w -s -X main.version=$(VERSION) -X main.commit=$(COMMIT)

.PHONY: build run clean

build:
	go build -o eval_loop -ldflags="$(LDFLAGS)" -trimpath .

install: build
	cp eval_loop ~/.local/bin/eval_loop

run:
	go run -ldflags="$(LDFLAGS)" .

clean:
	rm -f eval_loop

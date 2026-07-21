VERSION := $(shell git describe --tags --always 2>/dev/null || echo "dev")
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS := -w -s -X main.version=$(VERSION) -X main.commit=$(COMMIT)

.PHONY: build run clean

build:
	go build -o hone -ldflags="$(LDFLAGS)" -trimpath .

install: build
	cp hone ~/.local/bin/hone

run:
	go run -ldflags="$(LDFLAGS)" .

clean:
	rm -f hone

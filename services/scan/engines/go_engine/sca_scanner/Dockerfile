FROM golang:1.21-alpine as builder
WORKDIR /src
COPY . /src
RUN --mount=type=cache,target=/go/pkg/mod     cd services/scan/go_scanners &&     go build -o /out/sca-go ./sca_scanner

FROM alpine:3.19
COPY --from=builder /out/sca-go /usr/local/bin/sca-go
ENV SCAN_RESULTS_QUEUE=SCAN_RESULTS
CMD ["/usr/local/bin/sca-go"]

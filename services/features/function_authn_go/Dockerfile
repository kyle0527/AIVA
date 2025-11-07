FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . /app
RUN cd services/features/function_authn_go && go build -o /authn_worker ./cmd/worker

FROM alpine:3.19
COPY --from=builder /authn_worker /usr/local/bin/authn_worker
CMD ["/usr/local/bin/authn_worker"]

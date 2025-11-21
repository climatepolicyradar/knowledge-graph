# Multi-stage build for Next.js 15 application
# Stage 1: Builder - use full node image (has all build tools)
FROM node:20 AS builder

ARG TARGETOS
ARG TARGETARCH
WORKDIR /app

# Set environment variables for cross-platform build
ENV npm_config_target_arch=x64
ENV npm_config_target_platform=linux

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Ensure platform-specific lightningcss binary is present for Linux builds
RUN if [ "${TARGETOS}" = "linux" ]; then \
      case "${TARGETARCH}" in \
        "amd64") \
          LIGHTNINGCSS_VARIANT="lightningcss-linux-x64-gnu"; \
          ;; \
        "arm64") \
          LIGHTNINGCSS_VARIANT="lightningcss-linux-arm64-gnu"; \
          ;; \
        "arm") \
          LIGHTNINGCSS_VARIANT="lightningcss-linux-arm-gnueabihf"; \
          ;; \
        *) \
          LIGHTNINGCSS_VARIANT=""; \
          ;; \
      esac; \
      if [ -n "${LIGHTNINGCSS_VARIANT}" ]; then \
        LIGHTNINGCSS_VERSION=$(LIGHTNINGCSS_VARIANT="${LIGHTNINGCSS_VARIANT}" node -p "require('./package-lock.json').packages['node_modules/lightningcss'].optionalDependencies[process.env.LIGHTNINGCSS_VARIANT]"); \
        if [ -n "${LIGHTNINGCSS_VERSION}" ] && [ "${LIGHTNINGCSS_VERSION}" != "undefined" ]; then \
          npm_config_package_lock=false npm install --no-save "${LIGHTNINGCSS_VARIANT}@${LIGHTNINGCSS_VERSION}"; \
        fi; \
      fi; \
    fi

# Copy application code
COPY . .

# Ensure public directory exists (Next.js expects it during asset copy)
RUN mkdir -p public

# Rebuild native modules for target platform
RUN npm rebuild

# Build the Next.js application
RUN npm run build

# Stage 2: Production runtime
FROM node:20-slim

WORKDIR /app

# Install dumb-init for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends dumb-init && rm -rf /var/lib/apt/lists/*

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Create non-root user
RUN groupadd -g 1001 nodejs && useradd -u 1001 -g nodejs nextjs && chown -R nextjs:nodejs /app
USER nextjs

EXPOSE 3000

ENTRYPOINT ["dumb-init", "--"]
CMD ["npm", "start"]

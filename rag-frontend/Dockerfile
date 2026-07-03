# Stage 1: Build the React application using Vite
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application for production
RUN npm run build

# Stage 2: Serve the built application using Nginx
FROM nginx:alpine

# Copy the built assets from the builder stage to Nginx's default public directory
COPY --from=builder /app/dist /usr/share/nginx/html

# We need to configure Nginx to route all requests to index.html (for React Router)
# Nginx default configuration already exists, but we can override default.conf if needed.
# A simple way to support SPA routing is to replace default.conf
RUN echo "server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files \$uri \$uri/ /index.html; \
    } \
}" > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

# Use Nginx to serve static files
FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Add custom config
COPY nginx.conf /etc/nginx/conf.d

# Copy your built files (make sure they are in a folder like /build or /dist)
COPY ./build /usr/share/nginx/html

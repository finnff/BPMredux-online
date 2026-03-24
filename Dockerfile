FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy application files
COPY index.html /usr/share/nginx/html/
COPY src/ /usr/share/nginx/html/src/

# Expose port
EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]

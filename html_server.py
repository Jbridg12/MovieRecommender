#   file:   html_server.py
#   description:
#       this file allows content in this directory to be served over HTTP.
#       you should not modify this file in any capacity for Project 2.

import http.server
import socketserver

# define host, port, and handler
hostName = "localhost"
serverPort = 8080
Handler = http.server.SimpleHTTPRequestHandler

# create a simple TCP server that serves the directory directly to HTTP
with socketserver.TCPServer(("", serverPort), Handler) as webServer:
    print("[SERVING] HTML Server : http://%s:%s" % (hostName, serverPort))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    # close the server after we've hit a keyboard interrupt
    webServer.server_close()
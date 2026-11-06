import pgserver, time
srv = pgserver.get_server("/tmp/cardledger_pgdata")
print("URI:", srv.get_uri(), flush=True)
print(srv.psql("SELECT version()"), flush=True)
while True:
    time.sleep(60)

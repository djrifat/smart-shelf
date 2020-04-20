import sched, time

s = sched.scheduler(time.time,time.sleep)

def test(sc):
    print("do something")
    s.enter(5, 1, test, (sc,))

s.enter(5,1,test, (s,))
s.run()

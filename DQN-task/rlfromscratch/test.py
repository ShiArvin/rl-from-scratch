eps_start,eps_end = 1,0.1
for i in range(100):
    e = max(eps_end, eps_start - i/50 * (eps_start-eps_end))
    print(e)

print(0.9/50)
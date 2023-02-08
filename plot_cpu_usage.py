file = open("cpu_profile")
text = file.read()
stat_arr = text.split("###")

def calculate_cpu_percentage(prev_total, prev_idle, total, idle):
    diff_idle = idle - prev_idle
    diff_total = total - prev_total
    cpu_used = diff_total-diff_idle
    cpu_usage = (cpu_used/diff_total)*100
    return cpu_usage

cpu_percentages = {}
prev_total = {}
prev_idle = {}

for stat in stat_arr:
    lines = stat.split('\n')
    for line in lines:
        if line.startswith('cpu'):
            cpu_info = line.split()
            total = int(cpu_info[1]) + int(cpu_info[2]) + int(cpu_info[3]) + int(cpu_info[4]) + int(cpu_info[5]) + int(cpu_info[6]) + int(cpu_info[7])
            idle = int(cpu_info[4])
            if line[3]:
                key = line[3]
            else:
                key = "average"

            if cpu_percentages.get(key, None) == None:
                cpu_percentages[key] = []
                prev_total[key] = [total]
                prev_idle[key] = [idle]

            else:
                value = calculate_cpu_percentage(prev_total[key][-1], prev_idle[key][-1], total, idle)
                cpu_percentages[key].append(value)
                prev_total[key].append(total)
                prev_idle[key].append(idle)

print(cpu_percentages)

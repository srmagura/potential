import json
import datetime
import sys

M = 7
prec_str = '{:.5}'

#problem_list = ('trace-hat', 'trace-parabola', 'trace-line-sine',)
problem_list = ('trace-line-sine',)
boundary_list = ('arc', 'outer-sine', 'inner-sine', 'cubic', 'sine7',)

def create_header(max_error):
    global out
    out += '<tr><th></th>'
    for m in range(1, M+1):
        out += '<th>a{}</th>'.format(m)

    if max_error:
        out += '<th>Max error (1..7)</th>'
    out += '</tr>'

def create_row(label, array):
    global out
    out += '<tr><td class="label">{}</td>'.format(label)

    for num in array:
        out += '<td class="num">{}</td>'.format(prec_str.format(num))

    out += '</tr>'


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'acoef.dat'

all_data = json.load(open(filename))

title = 'calc_a_coef results ' + str(datetime.date.today())

out = ''
out += '<html><head><title>{}</title>'.format(title)
out += '''<style>
body {
    font-family: sans-serif;
}

table {
    border-collapse: collapse;
}

td {
    padding: 6px;
    margin: 0;
    border: 1px solid gray;
    width: 90px;
    text-align: center;
}

.problem {
    color: lightsalmon;
    font-size: 28pt;
}

.boundary {
    color: mediumslateblue;
    font-size: 20pt;
}
</style>
</head><body>'''
out += title

for problem in problem_list:
    problem_data = all_data[problem]
    out += '<br><span class="problem">{}</span> '.format(problem)
    out += 'k={} &nbsp;&nbsp;&nbsp;&nbsp;'.format(problem_data.pop('k'))
    out += ('error due to finite series, on arc = {}'.
        format(prec_str.format(problem_data['series_error'])))
    out += '<table>'
    create_header(False)
    create_row('expected', problem_data['arc']['fft_a_coef'])
    out += '</table>'

    for boundary in boundary_list:
        data = problem_data[boundary]
        out += '<br><span class="boundary">{}</span> '.format(boundary)
        out += 'beta={} &nbsp;&nbsp; '.format(prec_str.format(data['bet']))
        out += 'best_m1={}'.format(data['m1'])

        out += '<table>'
        create_header(True)
        create_row('error', data['error'] + [data['error7']])
        out += '</table>'

out += '</body></html>'

outfile_path = '/Users/sam/Google Drive/research/output/acoef_report.html'
outfile = open(outfile_path, 'w')
outfile.write(out)
print('Wrote {}'.format(outfile_path))

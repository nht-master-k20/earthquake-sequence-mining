// API base URL
const API_BASE = 'http://127.0.0.1:8386/api';

let dataTable;
let magChart, depthChart, monthChart;

$(document).ready(function() {
    initPage();
});

function initPage() {
    populateYearSelector();
}

async function populateYearSelector() {
    const yearSelect = $('#yearSelect');

    try {
        const response = await fetch(API_BASE + '/years');
        const result = await response.json();
        const years = result.years || [];

        yearSelect.empty();
        yearSelect.append('<option value="" disabled selected>-- Select Year --</option>');
        years.sort().forEach(year => {
            yearSelect.append('<option value="' + year + '">' + year + '</option>');
        });
        yearSelect.prop('disabled', false);
    } catch (error) {
        console.error('Error loading years:', error);
        yearSelect.prop('disabled', true);
    }
}

async function loadTable(year) {
    if (!year) {
        $('#dataTable').hide();
        $('#chartsSection').hide();
        $('#placeholderMsg').show();
        $('#tablePlaceholder').show();
        return;
    }

    $('#placeholderMsg').hide();
    $('#tablePlaceholder').hide();

    try {
        const response = await fetch(API_BASE + '/data/' + year);
        const result = await response.json();

        displayStats(result.stats);
        displayCharts(result.charts);
        displayTable(result.data);
    } catch (error) {
        console.error('Error loading data:', error);
        $('#placeholderMsg').html('<div class="alert alert-danger py-2">Failed to load data</div>');
        $('#placeholderMsg').show();
    }
}

function displayStats(stats) {
    $('#statCount').text(stats.total_events.toLocaleString());
    $('#statAvgMag').text(stats.avg_mag || '-');
    $('#statMaxMag').text(stats.max_mag || '-');
    $('#statAvgDepth').text(stats.avg_depth || '-');
}

function displayCharts(charts) {
    $('#chartsSection').show();

    if (magChart) magChart.dispose();
    magChart = echarts.init(document.getElementById('magChart'));
    magChart.setOption({
        tooltip: { trigger: 'item' },
        xAxis: { type: 'category', data: Object.keys(charts.mag_ranges) },
        yAxis: { type: 'value' },
        series: [{
            type: 'bar',
            data: Object.values(charts.mag_ranges),
            itemStyle: { color: '#667eea' }
        }]
    });

    if (depthChart) depthChart.dispose();
    depthChart = echarts.init(document.getElementById('depthChart'));
    depthChart.setOption({
        tooltip: { trigger: 'item' },
        xAxis: { type: 'category', data: Object.keys(charts.depth_ranges) },
        yAxis: { type: 'value' },
        series: [{
            type: 'bar',
            data: Object.values(charts.depth_ranges),
            itemStyle: { color: '#f5576c' }
        }]
    });

    if (monthChart) monthChart.dispose();
    monthChart = echarts.init(document.getElementById('monthChart'));
    monthChart.setOption({
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        },
        yAxis: { type: 'value' },
        series: [{
            type: 'line',
            data: charts.month_counts,
            smooth: true,
            itemStyle: { color: '#764ba2' },
            areaStyle: { color: 'rgba(118,75,162,0.3)' }
        }]
    });
}

function displayTable(data) {
    if ($.fn.DataTable.isDataTable('#dataTable')) {
        $('#dataTable').DataTable().destroy();
    }

    $('#dataTable').show();
    dataTable = $('#dataTable').DataTable({
        data: data.map(d => [
            d.time,
            d.place,
            d.mag,
            d.depth,
            d.lat,
            d.lon
        ]),
        columnDefs: [
            {targets: 0, width: '50px', title: '#', render: (d, t, r, m) => m.row + 1},
            {targets: 1, width: '180px', title: 'Time'},
            {targets: 2, title: 'Location'},
            {targets: 3, width: '70px', title: 'Magnitude'},
            {targets: 4, width: '120px', title: 'Depth (km)'},
            {targets: 5, width: '80px', title: 'Latitude'},
            {targets: 6, width: '80px', title: 'Longitude'}
        ],
        columns: [null, {data: 0}, {data: 1}, {data: 2}, {data: 3}, {data: 4}, {data: 5}],
        paging: false,
        order: [],
        dom: 'rtip',
        responsive: true,
        language: {
            info: 'Showing _START_ to _END_ of _TOTAL_ events'
        }
    });
}

$('#yearSelect').on('change', function() {
    const year = $(this).val();
    loadTable(year);
});

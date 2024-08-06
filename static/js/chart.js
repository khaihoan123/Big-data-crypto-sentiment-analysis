var ctx = document.getElementById('myChart').getContext('2d');
var ctx_2 = document.getElementById('sentimenChart').getContext('2d');


var chart = new Chart(ctx, {
  type: 'candlestick',
  data: {
    datasets: [{
      label: 'BTCUSD Price',
      data: barData,
    }, 
    {
      label: 'Model',
      type: 'line',
      data: line_data,
    //   hidden: true,
      backgroundColor: 'blue'
    },
    {
        label: 'Prediction',
        type: 'line',
        data: future_data, 
        // hidden: ,
        backgroundColor: 'red'
      }
]
  }
});


var chart2 = new Chart(ctx_2, {
    type: 'bar',
    data: {
      datasets: [{
        data: sentiment_data,
      },]
    },
    options: {
        plugins:{
            legend: {
             display: false
            }
           },
        title: {
            display: false,
          },
        scales: {
          x: {
            type: 'time',
            display: false,
            time: {
              unit: 'minute',
              displayFormats: {
                  minute: 'DD T'
              },
              tooltipFormat: 'DD T'
            },
            title: {
                display: false,
                // text: 'Sentimet'
              }
          },
          y: {
            title: {
              display: true,
              text: 'Sentimet'
            }
          }
        }
      }
  });




var update = function() {
  var dataset = chart.config.data.datasets[0];

  // color
  var colorScheme = document.getElementById('color-scheme').value;
  if (colorScheme === 'neon') {
    chart.config.data.datasets[0].backgroundColors = {
      up: '#01ff01',
      down: '#fe0000',
      unchanged: '#999',
    };
  } else {
    delete chart.config.data.datasets[0].backgroundColors;
  }

  // border
  var border = document.getElementById('border').value;
  if (border === 'false') {
    dataset.borderColors = 'rgba(0, 0, 0, 0)';  
  } else {
    delete dataset.borderColors;
  }

  // mixed charts
  var mixed = document.getElementById('mixed').value;
  if (mixed === 'true') {
    chart.config.data.datasets[1].hidden = false;
  } else {
    chart.config.data.datasets[1].hidden = true;
  }

  chart.update();
};

[...document.getElementsByTagName('select')].forEach(element => element.addEventListener('change', update));

document.getElementById('update').addEventListener('click', update);

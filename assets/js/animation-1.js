import { select } from 'd3-selection'
function kde(kernel, thresholds, data) {
    return thresholds.map(t => [t, d3.mean(data, d => kernel(t - d))]);
}
function epanechnikov(bandwidth) {
    return x => Math.abs(x /= bandwidth) <= 1 ? 0.75 * (1 - x * x) / bandwidth : 0;
}

line = d3.line()
    .curve(d3.curveBasis)
    .x(d => x(d[0]))
    .y(d => y(d[1]))

const promise = d3.json("/assets/faithful.json").then( function(data) {
    console.log(data)

    height = 500
    width = 954
    margin = ({top: 20, right: 30, bottom: 30, left: 40})

    x = d3.scaleLinear()
        .domain(d3.extent(data)).nice()
        .range([margin.left, width - margin.right])

    thresholds = x.ticks(40)

    density = kde(epanechnikov(1), thresholds, data)
    bins = d3.histogram()
            .domain(x.domain())
            .thresholds(thresholds)
        (data)
    
    console.log(bins, height, width, thresholds, density)


    y = d3.scaleLinear()
        .domain([0, d3.max(bins, d => d.length) / data.length])
        .range([height - margin.bottom, margin.top])
    
    xAxis = g => g
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x))
        .call(g => g.append("text")
            .attr("x", width - margin.right)
            .attr("y", -6)
            .attr("fill", "#000")
            .attr("text-anchor", "end")
            .attr("font-weight", "bold")
            .text(data.title))
    
    yAxis = g => g
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(null, "%"))
        .call(g => g.select(".domain").remove())
    
    var svg = d3.select("div#draw_here").append("svg")
        .attr("viewBox", [0, 0, width, height]);
    
    svg.append("g")
        .attr("fill", "#bbb")
        .selectAll("rect")
        .data(bins)
        .enter()
        .append("rect")
        .attr("x", d => x(d.x0) + 1)
        .attr("y", d => y(d.length / data.length))
        .attr("width", d => x(d.x1) - x(d.x0) - 1)
        .attr("height", d => y(0) - y(d.length / data.length));

    // line = d3.line()
    //     .curve(d3.curveBasis)
    //     .x(d => x(d[0]))
    //     .y(d => y(d[1]))
    
    svg.append("path")
        .data([density])
        .attr("id", "p1")
        .attr("fill", "none")
        .attr("stroke", "#000")
        .attr("stroke-width", 1.5)
        .attr("stroke-linejoin", "round")
        .attr("d", line);

    svg.append("g")
        .call(xAxis);

    svg.append("g")
        .call(yAxis);

    return data
});

function clicked() {
    promise.then(function(data) {
        var svg = d3.select("svg") // .transition();
        density = kde(epanechnikov(7), thresholds, data)
        console.log(density)

        console.log(svg.select("#p1"))

        svg.select("#p1")
            .transition()
            .data([density])
            // .attr("fill", "none")
            // .attr("stroke", "#000")
            // .attr("d", line)
            .attr("stroke-linejoin", "round");
        
        // duration(750).datum(density)
        // console.log(p)


        // p.duration(750).datum(density);
    });
}



// async function myFetch() {
//     let response = await fetch('/assets/faithful.json');

//     if (!response.ok) {
//         console.log(`HTTP error! status: ${response.status}`);
//     }

//     let data = await response.json();
//     data = Object.assign(data, {title: "Time between eruptions (min.)"})
//     console.log(data)

//     height = 500
//     width = 954
//     margin = ({top: 20, right: 30, bottom: 30, left: 40})

//     x = d3.scaleLinear()
//         .domain(d3.extent(data)).nice()
//         .range([margin.left, width - margin.right])

//     thresholds = x.ticks(40)

//     density = kde(epanechnikov(11), thresholds, data)
//     bins = d3.histogram()
//         .domain(x.domain())
//         .thresholds(thresholds)
//     (data)

//     y = d3.scaleLinear()
//         .domain([0, d3.max(bins, d => d.length) / data.length])
//         .range([height - margin.bottom, margin.top])
    
//     xAxis = g => g
//         .attr("transform", `translate(0,${height - margin.bottom})`)
//         .call(d3.axisBottom(x))
//         .call(g => g.append("text")
//             .attr("x", width - margin.right)
//             .attr("y", -6)
//             .attr("fill", "#000")
//             .attr("text-anchor", "end")
//             .attr("font-weight", "bold")
//             .text(data.title))
    
//     yAxis = g => g
//         .attr("transform", `translate(${margin.left},0)`)
//         .call(d3.axisLeft(y).ticks(null, "%"))
//         .call(g => g.select(".domain").remove())
    
    
//     var svg = d3.select("div#draw_here").append("svg")
//         .attr("viewBox", [0, 0, width, height]);
    
//     // d3.create("svg")

//     svg.append("path")
//         .datum(density)
//         .attr("fill", "none")
//         .attr("stroke", "#000")
//         .attr("stroke-width", 1.5)
//         .attr("stroke-linejoin", "round")
//         .attr("d", line);

//     svg.append("g")
//         .call(xAxis);

//     svg.append("g")
//         .call(yAxis);

//     svg.append("g")
//         .attr("fill", "#bbb")
//         .selectAll("rect")
//         .data(bins)
//         .join("rect")
//         .attr("x", d => x(d.x0) + 1)
//         .attr("y", d => y(d.length / data.length))
//         .attr("width", d => x(d.x1) - x(d.x0) - 1)
//         .attr("height", d => y(0) - y(d.length / data.length));

//     console.log(svg)
// }

// myFetch()
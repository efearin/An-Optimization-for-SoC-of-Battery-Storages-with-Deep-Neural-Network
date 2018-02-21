import simplex, simplestrategy

simplex_point = simplex.main([10, 10, 10], [10, 10, 10], [2,2,2], 1)
simple_point = simplestrategy.main([10, 10, 10], [10, 10, 10], [2,2,2], 1)
print('\nerror: '+ str(abs(simple_point.cost-simplex_point.cost)))
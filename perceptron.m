function [ avg_iter,avg_perror ] = perceptron( N,num_runs )
%perceptron runs perceptron learning algorithm
%   input: N:- no of data sets
%          num_runs:- number of runs
%   output:conv_iter:- iterations where the results converge
%          perror:- Prob[f(x)!= g(x)]
%Global Init
sum_iter = 0;
perror = 0;

for run=1:num_runs
    %Initialization
    conv_iter = 0;
    max_iter = 100000;
    %Generate 2 points in [-1,1] range
    point1 = -1 + 2 *rand(2,1);

    point2 = -1 + 2 *rand(2,1);

    x = [point1(1,1),point2(1,1)];

    y = [point1(2,1),point2(2,1)];

    plot(x,y);
    
    hold on;

    slope = (point1(2,1)-point2(2,1))/(point1(1,1)-point2(1,1));

    const = point1(2,1)-(slope*point1(1,1));

    % Generated f(x) which is the line
    % line is slope*x+const - y

    % Generate N points and calculate y for each points
    tmp = ones(1,N);
    datatmp = -1 + 2* rand(2,N);

    Data = [tmp;datatmp];

    scatter(Data(2,:),Data(3,:));

    y = zeros(1,N);

    for i=1:N
        if(Data(3,i) > slope * Data(2,i)+ const)
            y(1,i)= 1;
        else
            y(1,i) = -1;
        end
    end

    %weight vector initialization
    w = zeros(3,1);

    while(conv_iter<=max_iter)

        hyp = sign(w' * Data);

        for i=1:N
            if(hyp(1,i)==0)
                hyp(1,i)=-1;
            end
        end

        hyp = hyp + y;

        %Exit conditions
        %return means all the points are classified
        miscl = (hyp==0);

        if(miscl == zeros(1,N))
            break;
        end


        %Find misclassified point
        %check for indices where values are 1 in miscl array

        index = find(miscl == 1);

        d= size(index);

        %Pickup a random misclassified point and apply PLA
        mis_point_idx = index(1,randi([1,d(1,2)],1,1));

        w = w + (y(mis_point_idx) .* Data(:,mis_point_idx));

        conv_iter=conv_iter+1;

    end

    hold off;

    if(conv_iter > max_iter)
        warning('The PLA did not converge in %d iterations',max_iter);
    end

    fx = [-const;-slope;1];
    perr_per_run = monte_carlo_pla(fx,w);
    
    perror = perror + perr_per_run;
    sum_iter = sum_iter + conv_iter;

end

avg_perror = perror/num_runs;
avg_iter = sum_iter/num_runs;

end


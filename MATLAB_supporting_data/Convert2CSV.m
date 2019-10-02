clear all
Data = load('image_Numbers.mat');
Full_Numbers = Data.Full_Addresses(:, 2);
csvwrite('images_Full_Numbers.csv', Full_Numbers);
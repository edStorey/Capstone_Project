clear all

load digitStruct.mat

Full_Addresses = cell(length(digitStruct), 2);
for i = 1:length(digitStruct)

    for a = 1:length(digitStruct(i).bbox)
       digit = digitStruct(i).bbox(a).label;
       if digit == 10
           digit = 0;
       end
       
       if a == 1
           Address = digit;
       else 
           Address = str2num(strcat(num2str(Address),num2str(digit)));
       end
        
    end
    
    Full_Addresses{i, 1} = digitStruct(i).name;
    Full_Addresses{i, 2} = Address;
   
end

save('image_Numbers', 'Full_Addresses');